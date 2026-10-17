# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Keeps the translated documentation pages in step with the English ones.

The English pages are docs/<page>.html. A translation of one lives in its
language's folder, the one that holds the language's catalog (see README.md in
this folder), as src/resources/languages/<language>/docs/<page>.html, such as
src/resources/languages/zh_CN/docs/quickstart.html. It is the English page with
the readable text translated and nothing else changed: the same ids, links,
<code> snippets, styles and script logic. This tool checks that for every
translated page of every language, and tells which sections of an English page
changed after its translation was last reviewed. A language needs only its
folder: adding one needs no change in docs/ or here.

Each translated section carries data-source-sha256, the fingerprint of its
English section as it was when the translation was reviewed. The fingerprint is
taken over the section's prose and code only: its text with whitespace
collapsed, the text of <code> and command lines, and the translatable
attributes. Tags, classes and styles are not in it, so restyling the English
page makes no translation stale (the two pages' tags are compared with each
other instead, below). After the English page's text changes, `check` names the
sections whose fingerprint no longer matches; bring each up to date in the
translated page, then `stamp` it. A section is each element with an id that
sits beside the page's <section>s (a card, a divider, the header). A page
written without <section>s, as a flow of cards under dividers, has each
element with an id in its content block start a section, which runs on to the
next such element and takes the id-less elements before it (the page header,
a note, a footer). The sections also include each other child of <body> outside
that block (the top bar, the sidebar, the inline script) and the <title>.

    python src/resources/languages/Docs_Translation.py                    # what needs an update
    python src/resources/languages/Docs_Translation.py show cmd-label     # the English and translated section
    python src/resources/languages/Docs_Translation.py stamp cmd-label    # after translating that section
    python src/resources/languages/Docs_Translation.py stamp --all        # every section listed as stale
    python src/resources/languages/Docs_Translation.py generate           # the parts made from the language folders
    python src/resources/languages/Docs_Translation.py generate --check   # change nothing; fail if they are stale

Options: --page NAME and --language CODE narrow `check`, `show` and `stamp` to one
page or one language (the default is every page that has a translation, in
every language that has one; for `show` and `stamp` the section ids name the
page when only one has them). `check` exits with 1 while anything is left.

`generate` writes what depends on which languages exist:
  - each page's language dropdown: the <option>s between <!-- languages:begin -->
    and <!-- languages:end -->, one per language that holds the page, English
    first, each named as its language.json says, with the page's own selected and
    each value the relative address of that language's page;
  - each translated language's docs/language.css, the @font-face rules for the
    language's own font as its language.json names it (font.files under
    src/resources/fonts/desktop, font.web_range), written as the Viewer's pages
    get theirs (utilities/Language_Packs.web_font_css). A page may not hold a
    <style>, so the translated pages link this file.
`generate --check` changes nothing and exits with 1 if either is out of date.

The tool uses only the standard library and never changes the English page,
except the language dropdown that `generate` rewrites on it.

What `check` compares, apart from the fingerprints:
  - the same sections in the same order, and the same tags, ids, classes, event
    handlers and links around and inside them (a link to a file is compared by
    the file it names; a link to another documentation page, which lies in
    docs/ for English and in the language's docs/ folder for a translation,
    by the page's name, so the Chinese Quick Start's link to
    list_of_commands.html matches the English page's);
  - the same text in every <code>, in the command names and syntax boxes, in
    the example lines and in the sidebar's command list (VERBATIM_RULES): what
    readers type is never translated;
  - the same <style> or the same stylesheet links (the two pages are compared
    with each other, never with fixed contents; a translated page adds one more,
    language.css, when its language has a font), and the same script apart from
    the strings shown to readers (TRANSLATED_SCRIPT_STRINGS), each of which
    must be translated; the theme and sidebar code is the shared assets/docs.js,
    which shows the labels its button carries (TRANSLATED_ATTRIBUTES), and those
    too must be translated;
  - the language switch, a dropdown (<div class="lang-switch"> with a <select>
    in the top bar) that lists the page's languages, English first and then each
    language folder that holds the page, the same on every page: each option
    names its page, which exists, and the page's own language is the selected
    one. The select has an accessible label. The options sit between the
    comments <!-- languages:begin --> and <!-- languages:end -->, the block
    `generate` rewrites, and the shared docs/assets/docs.js opens the chosen page
    at the reader's place.
The translatable attributes (title, placeholder, aria-label) must be present
but may differ.
"""

import argparse
import hashlib
import html.parser
import json
import os
import posixpath
import re
import sys
from pathlib import Path

LANGUAGES_DIR = Path(__file__).resolve().parent
SRC_DIR = LANGUAGES_DIR.parents[1]
ROOT = SRC_DIR.parent
DOCS = ROOT / "docs"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities.Language_Packs import (  # noqa: E402
    DOCS_FOLDER,
    LANGUAGE_CODE,
    ManifestError,
    exact_file,
    exact_path,
    language_folders,
    read_pack,
    web_font_css,
)

STAMP = "data-source-sha256"
LANGUAGE_CSS = "language.css"
FONTS_DIR = SRC_DIR / "resources" / "fonts" / "desktop"

# Strings in the page's script that readers see: they differ between the pages,
# and the translated page must not keep the English one. Any other string in the
# script is code and must be the same.
TRANSLATED_SCRIPT_STRINGS = ("Copied!",)

# Attributes whose text docs/assets/docs.js shows to readers: the theme button's label for
# each mode, which the shared script reads from the page, so each page writes it in its
# own language. They differ between the pages, and the translated page must not keep the
# English text.
TRANSLATED_ATTRIBUTES = ("data-label-dark", "data-label-light")

# The comments around the language dropdown's <option>s: everything between them is the
# block that lists the languages, written the same way on every page.
LANGUAGES_BEGIN = "<!-- languages:begin -->"
LANGUAGES_END = "<!-- languages:end -->"

# Attributes whose text is for readers: present on both pages, free to differ.
TRANSLATABLE_ATTRIBUTES = ("title", "placeholder", "aria-label", "alt") + TRANSLATED_ATTRIBUTES

# The characters a language's font is asked for on a page: the unicode-range of
# Simplified Chinese's font in its language.json (font.web_range). Everything else on a
# page comes from the Latin faces or the system. cjk_characters lists them for a test
# that checks the font has each.
CJK_RANGES = (
    (0x3000, 0x303F), (0x3040, 0x30FF), (0x3100, 0x312F), (0x3200, 0x33FF),
    (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF),
)
CJK_PATTERN = re.compile("[" + "".join("%s-%s" % (chr(low), chr(high)) for low, high in CJK_RANGES) + "]")

VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
}

USAGE = (
    "python src/resources/languages/Docs_Translation.py              lists what needs an update\n"
    "python src/resources/languages/Docs_Translation.py show KEY     prints the English and the translated section\n"
    "python src/resources/languages/Docs_Translation.py stamp KEY... after reviewing those sections (or: stamp --all)"
)


class Element:
    """One tag of a page: its attributes, children (Element or str) and where it sits in the source."""

    def __init__(self, tag, attrs, start, start_tag, parent):
        self.tag = tag
        self.attrs = attrs
        self.children = []
        self.start = start
        self.start_tag = start_tag
        self.end = None
        self.parent = parent

    @property
    def classes(self):
        return set(self.attrs.get("class", "").split())

    def elements(self):
        return [child for child in self.children if isinstance(child, Element)]

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Element):
                yield from child.walk()

    def text(self):
        return "".join(child if isinstance(child, str) else child.text() for child in self.children)


class _Parser(html.parser.HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.source = source
        self.line_starts = [0]
        for index, char in enumerate(source):
            if char == "\n":
                self.line_starts.append(index + 1)
        self.root = Element("#document", {}, 0, "", None)
        self.stack = [self.root]

    def position(self):
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def _open(self, tag, attrs):
        element = Element(
            tag,
            {name: (value if value is not None else "") for name, value in attrs},
            self.position(),
            self.get_starttag_text() or "",
            self.stack[-1],
        )
        self.stack[-1].children.append(element)
        return element

    def handle_starttag(self, tag, attrs):
        element = self._open(tag, attrs)
        if tag in VOID_TAGS:
            element.end = element.start + len(element.start_tag)
        else:
            self.stack.append(element)

    def handle_startendtag(self, tag, attrs):
        element = self._open(tag, attrs)
        element.end = element.start + len(element.start_tag)

    def handle_endtag(self, tag):
        for depth in range(len(self.stack) - 1, 0, -1):
            if self.stack[depth].tag == tag:
                close = self.source.find(">", self.position())
                for element in self.stack[depth:]:
                    element.end = close + 1
                del self.stack[depth:]
                return

    def handle_data(self, data):
        children = self.stack[-1].children
        if children and isinstance(children[-1], str):
            children[-1] += data
        else:
            children.append(data)


def parse(source):
    parser = _Parser(source)
    parser.feed(source)
    parser.close()
    for element in parser.stack[1:]:
        element.end = len(source)
    return parser.root


def read(path):
    return Path(path).read_bytes().decode("utf-8")


def collapse(text):
    return " ".join(text.split())


def first(element, tag):
    for found in element.walk():
        if found.tag == tag:
            return found
    return None


# ---- What is never translated -------------------------------------------


def _in_command_list(element):
    node = element.parent
    while node is not None:
        if node.tag == "ul" and re.fullmatch(r"menu-.*-cmds", node.attrs.get("id", "")):
            return True
        node = node.parent
    return False


# The elements whose text readers type or match, so it is never translated. Each
# rule must match something on the English page: when a restyle renames a class,
# `check` says which rule to update here rather than silently checking less.
VERBATIM_RULES = (
    ("<code>", lambda e: e.tag == "code"),
    ("class syntax-content (a command's syntax box)", lambda e: "syntax-content" in e.classes),
    ("class command-card-name (a command's name)", lambda e: "command-card-name" in e.classes),
    ("class expr-string (expression syntax in a heading)", lambda e: "expr-string" in e.classes),
    ("class expr-label (expression syntax in a heading)", lambda e: "expr-label" in e.classes),
    ("class expr-struct (expression syntax in a heading)", lambda e: "expr-struct" in e.classes),
    (
        "the <span> of an example line (inside class example-item)",
        lambda e: e.tag == "span" and e.parent is not None and "example-item" in e.parent.classes,
    ),
    (
        "a command name in the sidebar's lists (<ul id=\"menu-...-cmds\">)",
        lambda e: e.tag == "span" and _in_command_list(e),
    ),
)


def _is_verbatim(element):
    return any(rule(element) for _, rule in VERBATIM_RULES)


class Page:
    """A parsed page and its sections.

    units lists (key, element) for each section, in document order; members[key] lists
    the elements the section takes in (just the element, but for a page that is a flow of
    cards under dividers, the divider and the elements after it up to the next one).
    The element that comes first carries the section's id and its data-source-sha256.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.source = read(path)
        self.root = parse(self.source)
        self.html = first(self.root, "html")
        self.head = first(self.root, "head")
        self.body = first(self.root, "body")
        self.problems = []
        self.members = {}
        self.units = self._find_units()

    def _content_block(self):
        """(the block that holds the page's content, whether it holds <section>s), or (None, False)."""
        sections = [e for e in self.body.walk() if e.tag == "section"]
        if sections:
            return sections[0].parent, True
        node = first(self.body, "main")
        while node is not None and len(node.elements()) == 1:
            node = node.elements()[0]
        return (node, False) if node is not None else (None, False)

    def _find_units(self):
        """(key, element) for the page's sections, in document order.

        The <title>; every child with an id of the block that holds the <section>s
        (or, without <section>s, of the content block of <main>, each running on to the next
        child with an id); and every other child of <body> or of the blocks around that one
        (the top bar, the sidebar, the inline script), named by its id or its tag."""
        units = []
        if self.head is None or self.body is None or self.html is None:
            self.problems.append("%s has no <html>, <head> and <body>." % self.path.name)
            return units
        title = first(self.head, "title")
        if title is not None:
            units.append(("title", title))
            self.members["title"] = [title]
        container, has_sections = self._content_block()
        if container is None:
            self.problems.append("%s has no <section>s and no <main> block to find the page's content by." % self.path.name)
            return units
        around = set()
        node = container.parent
        while node is not None:
            around.add(id(node))
            node = node.parent
        used = {}

        def add(key, elements):
            used[key] = used.get(key, 0) + 1
            name = key if used[key] == 1 else "%s-%d" % (key, used[key])
            units.append((name, elements[0]))
            self.members[name] = list(elements)

        def add_content(block):
            if has_sections:
                for member in block.elements():
                    key = member.attrs.get("id")
                    if key:
                        add(key, [member])
                    else:
                        self.problems.append(
                            "%s: a <%s> beside the <section>s has no id (%s); give every such element an id so its "
                            "translation can be matched." % (self.path.name, member.tag, collapse(member.start_tag)[:70])
                        )
                return
            group = []
            for member in block.elements():
                if member.attrs.get("id") and group:
                    add(self._flow_key(group[0]), group)
                    group = []
                group.append(member)
            if group:
                add(self._flow_key(group[0]), group)

        def descend(node):
            for child in node.elements():
                if child is container:
                    add_content(child)
                elif id(child) in around:
                    descend(child)
                elif child.tag == "script" and "src" in child.attrs:
                    continue
                else:
                    add(child.attrs.get("id") or child.tag, [child])

        descend(self.body)
        keys = [key for key, _ in units]
        for key in sorted({key for key in keys if keys.count(key) > 1}):
            self.problems.append("%s: the section id '%s' is used more than once." % (self.path.name, key))
        return units

    @staticmethod
    def _flow_key(element):
        """The name of a section that starts with element: its id, else its first class, else its tag."""
        return element.attrs.get("id") or (element.attrs.get("class", "").split() or [element.tag])[0]

    def unit(self, key):
        for name, element in self.units:
            if name == key:
                return element
        return None

    def unit_members(self, key):
        return self.members.get(key, [])

    @property
    def keys(self):
        return [key for key, _ in self.units]

    def verbatim_rule_counts(self):
        """How many elements of this page each of VERBATIM_RULES matches."""
        return [sum(1 for element in self.root.walk() if rule(element)) for _, rule in VERBATIM_RULES]


# ---- Scripts ------------------------------------------------------------

_JS_TOKEN = re.compile(
    r"""(?P<comment>//[^\n]*|/\*.*?\*/)|(?P<string>'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"|`(?:\\.|[^`\\])*`)""",
    re.S,
)


def _script_text(element):
    lines = element.text().replace("\r\n", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def _script_parts(text):
    """(code with each string replaced by a marker, the strings in order)."""
    strings = []

    def take(match):
        if match.group("string") is None:
            return match.group(0)
        strings.append(match.group("string")[1:-1])
        return "\0S\0"

    return _JS_TOKEN.sub(take, text), strings


# ---- Fingerprints -------------------------------------------------------


def _skip_in_fingerprint(element):
    return "lang-switch" in element.classes


def _reads_as_a_sentence(text):
    """A script string that is words rather than a selector, an id or a value."""
    return bool(re.search(r"[A-Za-z]{2,}[ ,.:;!?']+[A-Za-z]{2,}", text)) and not re.search(r"[#.\[\]=>{}$()/\-]", text)


def canonical(element):
    """The section's prose and code with whitespace collapsed, as a list of tokens.

    Only what a translator reads goes in: text, code and command text, and the
    translatable attributes. Tags, classes, ids and styles do not, so the English
    page can be restyled without making its translation stale. For the script
    that is the strings readers see (TRANSLATED_SCRIPT_STRINGS) and those that
    read as sentences, so a new message in the script makes it stale."""
    if element.tag == "script":
        strings = _script_parts(_script_text(element))[1]
        return ["script strings"] + [s for s in strings if s in TRANSLATED_SCRIPT_STRINGS or _reads_as_a_sentence(s)]
    tokens = []

    def walk(node):
        if _skip_in_fingerprint(node):
            return
        for name in TRANSLATABLE_ATTRIBUTES:
            if name in node.attrs:
                tokens.append(["@" + name, collapse(node.attrs[name])])
        if _is_verbatim(node):
            tokens.append(["code", collapse(node.text())])
            return
        for child in node.children:
            if isinstance(child, str):
                text = collapse(child)
                if text:
                    tokens.append(text)
            else:
                walk(child)

    walk(element)
    return tokens


def _digest(tokens):
    data = json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def fingerprint(element):
    return _digest(canonical(element))


def section_fingerprint(page, key):
    """The fingerprint of a section: that of its element, or of all its elements in order."""
    members = page.unit_members(key)
    if len(members) == 1:
        return fingerprint(members[0])
    tokens = []
    for element in members:
        tokens.extend(canonical(element))
    return _digest(tokens)


# ---- Structure ----------------------------------------------------------


def _link_target(page_path, value):
    """A link, compared by what it names: a fragment or a URL as written, a file by its resolved path.

    A page of the documentation, which is in docs/ for English and in the language's own
    docs/ folder for a translation, is named by its file name alone, so a translated page's
    link to a sibling matches the English page's link to its sibling."""
    if not value or value.startswith("#") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) or value.startswith("//"):
        return value
    target, _, fragment = value.partition("#")
    resolved = posixpath.normpath(posixpath.join(Path(page_path).parent.as_posix(), target))
    if resolved.endswith(".html") and posixpath.basename(posixpath.dirname(resolved)) == DOCS_FOLDER:
        resolved = "docs-page:" + posixpath.basename(resolved)
    return resolved + ("#" + fragment if fragment else "")


def _attributes(page, element, is_html_root=False):
    shown = {}
    for name, value in element.attrs.items():
        if name == STAMP or (is_html_root and name == "lang"):
            continue
        if name in TRANSLATABLE_ATTRIBUTES:
            shown[name] = "<text>"
        elif name in ("href", "src"):
            shown[name] = _link_target(page.path, value)
        else:
            shown[name] = value
    return tuple(sorted(shown.items()))


def _skeleton(page, key):
    """(tags, texts that are never translated) of one section; the language switch is left out."""
    tokens, verbatim = [], []

    def walk(node, inside_verbatim=False):
        if _skip_in_fingerprint(node):
            return
        tokens.append(("<", node.tag, _attributes(page, node)))
        own = _is_verbatim(node) and not inside_verbatim
        if own:
            verbatim.append(collapse(node.text()))
        for child in node.children:
            if isinstance(child, Element):
                walk(child, inside_verbatim or own)
        tokens.append((">", node.tag))

    for element in page.unit_members(key):
        walk(element)
    return tokens, verbatim


def _frame(page, translated=False):
    """The page outside its sections: head, body, the blocks around them and the styles, with each section as a marker.

    A translated page's own link to language.css is left out: it is the one stylesheet
    link the English page does not have (see _language_css_problems)."""
    section_of = {id(element): key for key, element in page.units}
    taken_in = {id(element) for key, _ in page.units for element in page.unit_members(key)[1:]}
    tokens = []

    def walk(node):
        key = section_of.get(id(node))
        if key is not None:
            tokens.append(("section", key))
            return
        if id(node) in taken_in:
            return
        if translated and node.tag == "link" and node.attrs.get("href") == LANGUAGE_CSS:
            return
        if node.tag == "#document":
            for child in node.children:
                if isinstance(child, Element):
                    walk(child)
            return
        tokens.append(("<", node.tag, _attributes(page, node, is_html_root=node.tag == "html")))
        if node.tag == "style":
            tokens.append(("style text", _script_text(node)))
        for child in node.children:
            if isinstance(child, Element):
                walk(child)
            elif node.tag not in ("style", "script") and collapse(child):
                tokens.append(("text", collapse(child)))
        tokens.append((">", node.tag))

    walk(page.root)
    return tokens


def _first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def _show(text, limit=90):
    text = collapse(text)
    return "'%s'" % (text if len(text) <= limit else text[: limit - 3] + "...")


def _compare_script(key, english, translated, problems):
    english_code, english_strings = _script_parts(_script_text(english))
    translated_code, translated_strings = _script_parts(_script_text(translated))
    if english_code != translated_code or len(english_strings) != len(translated_strings):
        a, b = english_code.split("\n"), translated_code.split("\n")
        line = _first_difference(a, b)
        problems.append(
            "Section '%s' (the script): its code differs from the English page's at line %d of the script "
            "(English %s, translated %s). Only the strings in TRANSLATED_SCRIPT_STRINGS may differ; copy the "
            "English script and translate those strings."
            % (key, line + 1, _show(a[line] if line < len(a) else "(end)"), _show(b[line] if line < len(b) else "(end)"))
        )
        return
    for english_string, translated_string in zip(english_strings, translated_strings):
        if english_string in TRANSLATED_SCRIPT_STRINGS:
            if translated_string == english_string:
                problems.append(
                    "Section '%s' (the script): the string '%s' is shown to readers but is still English; translate it."
                    % (key, english_string)
                )
        elif english_string != translated_string:
            problems.append(
                "Section '%s' (the script): the string '%s' differs from the English page's '%s'. If readers see it, "
                "add it to TRANSLATED_SCRIPT_STRINGS in src/resources/languages/Docs_Translation.py; otherwise copy it unchanged."
                % (key, translated_string, english_string)
            )


def _tag_text(tokens, index):
    if index >= len(tokens):
        return "(the section ends)"
    kind, tag = tokens[index][0], tokens[index][1]
    attrs = "".join(' %s="%s"' % pair for pair in tokens[index][2]) if kind == "<" else ""
    return "<%s%s%s>" % ("/" if kind == ">" else "", tag, attrs)


def _compare_section(key, english_page, translated_page, problems):
    if key == "title":
        return
    english, translated = english_page.unit(key), translated_page.unit(key)
    if english.tag == "script":
        _compare_script(key, english, translated, problems)
        return
    english_tokens, english_verbatim = _skeleton(english_page, key)
    translated_tokens, translated_verbatim = _skeleton(translated_page, key)
    if english_tokens != translated_tokens:
        index = _first_difference(english_tokens, translated_tokens)
        problems.append(
            "Section '%s': the tags differ from the English page's at tag %d: English has %s, the translation has %s. "
            "Keep every tag, id, class and attribute as in the English page; only the text between them is translated."
            % (key, index + 1, _show(_tag_text(english_tokens, index), 140), _show(_tag_text(translated_tokens, index), 140))
        )
        return
    if english_verbatim != translated_verbatim:
        index = _first_difference(english_verbatim, translated_verbatim)
        problems.append(
            "Section '%s': the code and command text differs from the English page's (English has %d such snippets, "
            "the translation %d): snippet %d is %s in English and %s in the translation. <code> text, command names, "
            "syntax boxes and example lines are never translated."
            % (
                key, len(english_verbatim), len(translated_verbatim), index + 1,
                _show(english_verbatim[index]) if index < len(english_verbatim) else "missing",
                _show(translated_verbatim[index]) if index < len(translated_verbatim) else "missing",
            )
        )


def _frame_problem(english_page, translated_page):
    english_frame, translated_frame = _frame(english_page), _frame(translated_page, translated=True)
    if english_frame == translated_frame:
        return None
    index = _first_difference(english_frame, translated_frame)

    def show(frame):
        if index >= len(frame):
            return "(the page ends)"
        return _show(" ".join(str(part) for part in frame[index]), 160)

    return (
        "The page outside its sections (head, <style>, the blocks around them) differs from the English page's at "
        "item %d: English has %s, the translation has %s. Everything but the text of the sections must be "
        "identical, the <style> block or stylesheet link included." % (index + 1, show(english_frame), show(translated_frame))
    )


# ---- Which pages and languages exist ------------------------------------


def language_attribute(code):
    return code.replace("_", "-")


def english_pages(docs=DOCS):
    """The English pages: docs/<page>.html, as their names without the extension, sorted."""
    return sorted(path.stem for path in Path(docs).glob("*.html"))


def translation_path(code, page, languages_dir=LANGUAGES_DIR):
    """The language's translation of docs/<page>.html: <languages_dir>/<code>/docs/<page>.html, or None."""
    folder = exact_path(languages_dir, code, DOCS_FOLDER)
    return exact_file(folder, page + ".html") if folder is not None else None


def translations(docs=DOCS, languages_dir=LANGUAGES_DIR):
    """(page, language code, English path, translated path) of every translated page, by page and then language."""
    found = []
    for page in english_pages(docs):
        for code in language_folders(languages_dir):
            path = translation_path(code, page, languages_dir)
            if path is not None:
                found.append((page, code, Path(docs) / (page + ".html"), path))
    return found


def page_languages(english_path, languages_dir=LANGUAGES_DIR):
    """(code, page) of every language the page exists in: "en", then each language folder that has it."""
    english_path = Path(english_path)
    pages = [("en", english_path)]
    for code in language_folders(languages_dir):
        path = translation_path(code, english_path.stem, languages_dir)
        if path is not None:
            pages.append((code, path))
    return pages


def language_name(code, languages_dir=LANGUAGES_DIR):
    """What the dropdown calls the language: English, or the name its language.json gives, or its code."""
    if code == "en":
        return "English"
    try:
        return read_pack(code, languages_dir).name or code
    except ManifestError:
        return code


def _display(path):
    """A path as the messages name it: from the project folder when it is in it."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


# ---- The language switch ------------------------------------------------


def _language_switch_problems(page, language, siblings):
    """The switch on one page: every language listed, the page's own selected, each option reaching its page."""
    problems = []
    switch = next((e for e in page.root.walk() if "lang-switch" in e.classes), None)
    select = next((e for e in switch.walk() if e.tag == "select"), None) if switch is not None else None
    if select is None:
        return [
            '%s has no language switch (a <div class="lang-switch"> holding a <select> beside the theme button).'
            % page.path.name
        ]
    if not select.attrs.get("aria-label", "").strip():
        problems.append("%s: the language switch's <select> needs an aria-label naming it for screen readers." % page.path.name)
    options = [e for e in select.elements() if e.tag == "option"]
    # The options sit between two comments, the block `generate` rewrites from the language folders.
    segment = page.source[select.start:select.end]
    begin, end = segment.find(LANGUAGES_BEGIN), segment.find(LANGUAGES_END)
    if not (
        0 <= begin < end
        and all(begin < option.start - select.start < end for option in options)
    ):
        problems.append(
            "%s: the language switch's options must all sit between %s and %s, inside the <select>, so that the block "
            "can be regenerated." % (page.path.name, LANGUAGES_BEGIN, LANGUAGES_END)
        )
    expected = [lang for lang, _ in siblings]
    if [e.attrs.get("lang") for e in options] != expected:
        return problems + [
            "%s: the language switch should list %s in this order, by the lang attributes of its options; it has %s. "
            "Every page lists every language, English first and then one for each language folder that holds the page; "
            "run `python src/resources/languages/Docs_Translation.py generate` to write the block."
            % (page.path.name, expected, [e.attrs.get("lang") for e in options])
        ]
    for option, (lang, target) in zip(options, siblings):
        base = option.attrs.get("value", "").split("#")[0]
        resolved = (page.path.parent / base).resolve() if base else None
        if resolved is None or resolved != Path(target).resolve():
            problems.append(
                "%s: the switch's option for %s has the value '%s'; it should reach %s."
                % (page.path.name, lang, option.attrs.get("value", ""), _display(target))
            )
        elif not resolved.is_file():
            problems.append("%s: the switch's option for %s leads to %s, which does not exist." % (page.path.name, lang, resolved))
        if lang == language and "selected" not in option.attrs:
            problems.append(
                "%s: the switch must have %s, the page's own language, selected (give its <option> the attribute "
                "selected)." % (page.path.name, lang)
            )
        if lang != language and "selected" in option.attrs:
            problems.append("%s: the switch's option for %s must not be the selected one." % (page.path.name, lang))
    return problems


def _language_css_problems(page, code, languages_dir):
    """A translated page links its language's language.css when the language has a font, and only then."""
    try:
        font = read_pack(code, languages_dir).font
    except ManifestError as error:
        return ["The language's language.json can't be used: %s" % error]
    wants = font is not None and bool(font.web_range)
    links = [
        e for e in page.head.walk()
        if e.tag == "link" and e.attrs.get("href") == LANGUAGE_CSS and e.attrs.get("rel") == "stylesheet"
    ] if page.head is not None else []
    if wants and not links:
        return [
            '%s must link its language\'s font, <link href="%s" rel="stylesheet"> after the link to assets/docs.css: the '
            "page may not hold a <style> of its own, and the font is named by %s/language.json."
            % (page.path.name, LANGUAGE_CSS, code)
        ]
    if links and not wants:
        return [
            "%s links %s, but %s/language.json names no font with a web_range for it; take the link out."
            % (page.path.name, LANGUAGE_CSS, code)
        ]
    if len(links) > 1:
        return ["%s links %s more than once." % (page.path.name, LANGUAGE_CSS)]
    if links and not (page.path.parent / LANGUAGE_CSS).is_file():
        return [
            "%s links %s, which does not exist beside it; run `python src/resources/languages/Docs_Translation.py "
            "generate`." % (page.path.name, LANGUAGE_CSS)
        ]
    return []


def _sibling_link_problems(page, english_dir):
    """The links of a page to another documentation page reach that page in the page's own language.

    A translated page links the page of the same name beside it (the Chinese command reference's Quick
    Start tab goes to the Chinese Quick Start), or the English page when its language has none; an
    English page links the English one. The links are compared by page name between the pages
    (_link_target), so this is what keeps a translation from sending its reader back to English."""
    problems = []
    for element in page.root.walk():
        if element.tag != "a":
            continue
        base = element.attrs.get("href", "").partition("#")[0]
        if not base.endswith(".html") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", base) or base.startswith("//"):
            continue
        target = (page.path.parent / base).resolve()
        if target.parent.name != DOCS_FOLDER:
            continue
        own = page.path.parent / target.name
        wanted = own if own.is_file() else Path(english_dir) / target.name
        if target != wanted.resolve():
            problems.append(
                "%s: the link to '%s' reaches %s; it should reach the page in its own language, %s."
                % (page.path.name, element.attrs.get("href", ""), _display(target), _display(wanted))
            )
    return problems


# ---- The check ----------------------------------------------------------


class Report:
    def __init__(self, language, page_name, english_path=None, translated_path=None):
        self.language = language
        self.page_name = page_name
        self.english_path = english_path
        self.translated_path = translated_path
        self.problems = []  # the pages differ in more than text: fix before anything else
        self.stale = []  # sections whose English changed since the translation was reviewed
        self.unstamped = []  # sections without a fingerprint

    @property
    def ok(self):
        return not (self.problems or self.stale or self.unstamped)

    @property
    def english(self):
        return _display(self.english_path) if self.english_path is not None else "docs/%s.html" % self.page_name

    @property
    def translated(self):
        if self.translated_path is not None:
            return _display(self.translated_path)
        return "src/resources/languages/%s/docs/%s.html" % (self.language, self.page_name)

    def format(self):
        lines = []
        english, translated = self.english, self.translated
        if self.problems:
            lines.append("%s no longer matches %s:" % (translated, english))
            lines.extend("  - " + problem for problem in self.problems)
        if self.stale:
            lines.append(
                "The text of %s changed in %s after the translation was reviewed: %s."
                % ("these sections" if len(self.stale) > 1 else "this section", english, ", ".join(self.stale))
            )
            lines.append("  Update %s in %s, then record it as reviewed." % ("each" if len(self.stale) > 1 else "it", translated))
        if self.unstamped:
            lines.append("These sections of %s have no %s fingerprint: %s." % (translated, STAMP, ", ".join(self.unstamped)))
        if not self.ok:
            lines.append("How to bring the translation up to date (add --page %s --language %s to name this one):"
                         % (self.page_name, self.language))
            lines.extend("  " + line for line in USAGE.splitlines())
        return "\n".join(lines)


def check_page(english_path, translated_path, language=None):
    """Compares a translated page with its English page; returns a Report.

    The translated page is <languages folder>/<language>/docs/<page>.html; language,
    when not given, is the folder it is in."""
    english_path, translated_path = Path(english_path), Path(translated_path)
    languages_dir = translated_path.parents[2]
    language = language or translated_path.parents[1].name
    report = Report(language, english_path.stem, english_path, translated_path)
    english, translated = Page(english_path), Page(translated_path)
    report.problems.extend(english.problems)
    report.problems.extend(translated.problems)

    for name in TRANSLATED_ATTRIBUTES:
        english_values = [e.attrs[name] for e in english.root.walk() if name in e.attrs]
        translated_values = [e.attrs[name] for e in translated.root.walk() if name in e.attrs]
        if len(english_values) == len(translated_values):
            for english_value, translated_value in zip(english_values, translated_values):
                if collapse(english_value) == collapse(translated_value):
                    report.problems.append(
                        "The attribute %s=\"%s\" is shown to readers by docs/assets/docs.js but is still English in the "
                        "translation; translate it." % (name, english_value)
                    )

    expected_lang = language_attribute(language)
    if english.html is not None and english.html.attrs.get("lang") != "en":
        report.problems.append('%s should declare <html lang="en">.' % english_path.name)
    if translated.html is not None and translated.html.attrs.get("lang") != expected_lang:
        report.problems.append('%s should declare <html lang="%s">.' % (translated_path.name, expected_lang))

    english_keys, translated_keys = english.keys, translated.keys
    missing = [key for key in english_keys if key not in translated_keys]
    extra = [key for key in translated_keys if key not in english_keys]
    if missing:
        report.problems.append(
            "The translation lacks these sections of the English page: %s. Translate and add them in the same place."
            % ", ".join(missing)
        )
    if extra:
        report.problems.append(
            "The translation has sections the English page lacks: %s. Remove them, or add them to the English page."
            % ", ".join(extra)
        )
    if not missing and not extra and english_keys != translated_keys:
        report.problems.append(
            "The sections are in another order than in the English page. English: %s. Translation: %s."
            % (", ".join(english_keys), ", ".join(translated_keys))
        )

    shared = [key for key in english_keys if key in translated_keys]
    if not (missing or extra or english.problems or translated.problems):
        frame = _frame_problem(english, translated)
        if frame:
            report.problems.append(frame)
        else:
            for key in shared:
                _compare_section(key, english, translated, report.problems)

    report.problems.extend(_language_css_problems(translated, language, languages_dir))
    report.problems.extend(_sibling_link_problems(english, english_path.parent))
    report.problems.extend(_sibling_link_problems(translated, english_path.parent))
    siblings = [(language_attribute(code), path) for code, path in page_languages(english_path, languages_dir)]
    if translated_path.resolve() not in [path.resolve() for _, path in siblings]:
        siblings.append((expected_lang, translated_path))
    report.problems.extend(_language_switch_problems(english, "en", siblings))
    report.problems.extend(_language_switch_problems(translated, expected_lang, siblings))

    if not (missing or extra or english.problems or translated.problems):
        for key in shared:
            stamp = translated.unit(key).attrs.get(STAMP)
            if not stamp:
                report.unstamped.append(key)
            elif stamp != section_fingerprint(english, key):
                report.stale.append(key)
    return report


def unused_rules(docs=DOCS):
    """The entries of VERBATIM_RULES, TRANSLATED_SCRIPT_STRINGS and TRANSLATED_ATTRIBUTES that no English page uses.

    A restyle that renames a class or a script string would otherwise make the check look at
    less without saying so; each entry must be found on some English page. Returns the messages."""
    pages = [Page(Path(docs) / (name + ".html")) for name in english_pages(docs)]
    messages = []
    for index, (description, _) in enumerate(VERBATIM_RULES):
        if not any(page.verbatim_rule_counts()[index] for page in pages):
            messages.append(
                "VERBATIM_RULES in src/resources/languages/Docs_Translation.py has the rule '%s', which matches nothing on "
                "any English page any more. Update the rule to the pages' new markup, so the text it protects is still "
                "checked." % description
            )
    strings = set()
    for page in pages:
        for key, element in page.units:
            if element.tag == "script":
                strings.update(_script_parts(_script_text(element))[1])
    for text in TRANSLATED_SCRIPT_STRINGS:
        if text not in strings:
            messages.append(
                "TRANSLATED_SCRIPT_STRINGS in src/resources/languages/Docs_Translation.py lists '%s', which no English "
                "page's script has any more. List the strings the scripts show to readers now." % text
            )
    for name in TRANSLATED_ATTRIBUTES:
        if not any(name in element.attrs for page in pages for element in page.root.walk()):
            messages.append(
                "TRANSLATED_ATTRIBUTES in src/resources/languages/Docs_Translation.py lists '%s', which no element of any "
                "English page has any more. List the attributes the shared script shows to readers now." % name
            )
    return messages


def check_all(docs=DOCS, languages_dir=LANGUAGES_DIR, page=None, language=None):
    """A Report for each translated page (of the page and the language, when given)."""
    return [
        check_page(english, translated, code)
        for name, code, english, translated in translations(docs, languages_dir)
        if page in (None, name) and language in (None, code)
    ]


def english_fingerprints(english_path):
    page = Page(english_path)
    return {key: section_fingerprint(page, key) for key, _ in page.units}


def stamp_sections(english_path, translated_path, keys):
    """Records `keys` as reviewed: sets their data-source-sha256 to the English section's fingerprint."""
    english, translated = Page(english_path), Page(translated_path)
    wanted = list(dict.fromkeys(keys))
    unknown = [key for key in wanted if english.unit(key) is None or translated.unit(key) is None]
    if unknown:
        raise KeyError("no such section on both pages: %s" % ", ".join(unknown))
    source = translated.source
    edits = []
    for key in wanted:
        element = translated.unit(key)
        tag = element.start_tag
        value = '%s="%s"' % (STAMP, section_fingerprint(english, key))
        if re.search(r"\s%s\s*=" % STAMP, tag):
            pattern = r"(\s)%s\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)" % STAMP
            new_tag = re.sub(pattern, lambda match: match.group(1) + value, tag, count=1)
        else:
            close = "/>" if tag.endswith("/>") else ">"
            new_tag = tag[: -len(close)].rstrip() + " " + value + close
        edits.append((element.start, element.start + len(tag), new_tag))
    for start, end, new_tag in sorted(edits, reverse=True):
        source = source[:start] + new_tag + source[end:]
    Path(translated_path).write_bytes(source.encode("utf-8"))
    return wanted


def section_source(page, key):
    members = page.unit_members(key)
    return page.source[members[0].start : members[-1].end] if members else None


# ---- What is made from the language folders ------------------------------


def _relative(target, start):
    return os.path.relpath(str(target), str(start)).replace("\\", "/")


def language_options(page_path, english_path, languages_dir=LANGUAGES_DIR):
    """The <option> lines of the language dropdown for the page at page_path, one per language that holds the page."""
    page_dir = Path(os.path.abspath(page_path)).parent
    own = Path(os.path.abspath(page_path))
    options = []
    for code, path in page_languages(english_path, languages_dir):
        target = Path(os.path.abspath(path))
        options.append(
            '<option value="%s" lang="%s" data-lang="%s"%s>%s</option>'
            % (
                _relative(target, page_dir), language_attribute(code), code, " selected" if target == own else "",
                language_name(code, languages_dir),
            )
        )
    return options


def regenerated_page(page_path, english_path, languages_dir=LANGUAGES_DIR):
    """(the page's source with its language dropdown rewritten, or None if it has no block to rewrite)."""
    source = Path(page_path).read_bytes().decode("utf-8")
    begin, end = source.find(LANGUAGES_BEGIN), source.find(LANGUAGES_END)
    if begin < 0 or end < begin:
        return None
    newline = "\r\n" if "\r\n" in source else "\n"
    line_start = source.rfind("\n", 0, begin) + 1
    indent = source[line_start:begin]
    if indent.strip():
        indent = ""
    body = "".join(indent + option + newline for option in language_options(page_path, english_path, languages_dir))
    return source[: begin + len(LANGUAGES_BEGIN)] + newline + body + indent + source[end:]


def language_css(code, languages_dir=LANGUAGES_DIR):
    """The text of <code>/docs/language.css, or None when the language names no font for a page."""
    font = read_pack(code, languages_dir).font
    if font is None or not font.web_range:
        return None
    prefix = _relative(Path(os.path.abspath(Path(languages_dir).parent / "fonts" / "desktop")),
                       Path(os.path.abspath(Path(languages_dir) / code / DOCS_FOLDER))) + "/"
    return (
        "/* The font of this language's documentation pages: made from the \"font\" of ../language.json by\n"
        " * `python src/resources/languages/Docs_Translation.py generate`; do not edit it by hand. The rules are\n"
        " * the ones the Viewer's pages get (utilities/Language_Packs.web_font_css), for the same family name,\n"
        " * which docs/assets/docs.css names after 'Noto Sans' in its font stack; the bold face (700) is what\n"
        " * the pages' strong text uses in this language (docs.css --strong).\n"
        " */\n\n" + web_font_css(font, prefix) + "\n"
    )


def _normal(text):
    return text.replace("\r\n", "\n")


def generate(docs=DOCS, languages_dir=LANGUAGES_DIR, check=False):
    """Writes the language dropdown of every page and the language.css of every translated language.

    With check, changes nothing. Returns [(path, what is out of date or can't be made), ...]: empty
    when everything is as `generate` would write it."""
    stale = []
    for name in english_pages(docs):
        english_path = Path(docs) / (name + ".html")
        for code, path in page_languages(english_path, languages_dir):
            source = path.read_bytes().decode("utf-8")
            rewritten = regenerated_page(path, english_path, languages_dir)
            if rewritten is None:
                stale.append((path, "has no %s ... %s block for the language dropdown" % (LANGUAGES_BEGIN, LANGUAGES_END)))
            elif rewritten != source:
                stale.append((path, "its language dropdown does not list the languages that hold the page"))
                if not check:
                    path.write_bytes(rewritten.encode("utf-8"))
    for code in language_folders(languages_dir):
        folder = exact_path(languages_dir, code, DOCS_FOLDER)
        if folder is None:
            continue
        target = folder / LANGUAGE_CSS
        try:
            wanted = language_css(code, languages_dir)
        except ManifestError as error:
            stale.append((target, "the language.json can't be used: %s" % error))
            continue
        present = exact_file(folder, LANGUAGE_CSS)
        if wanted is None:
            if present is not None:
                stale.append((present, "the language names no font for a page, so the file is not wanted"))
                if not check:
                    present.unlink()
        elif present is None or _normal(present.read_bytes().decode("utf-8")) != _normal(wanted):
            stale.append((target, "it does not hold the font rules language.json gives"))
            if not check:
                # A checkout may hold the file with CRLF line ends (git's text=auto on Windows): keep them.
                crlf = present is not None and b"\r\n" in present.read_bytes()
                target.write_bytes((wanted.replace("\n", "\r\n") if crlf else wanted).encode("utf-8"))
    return stale


# ---- Command line -------------------------------------------------------


def cjk_characters(*paths):
    """The characters of the pages that a Chinese language font is asked for (CJK_RANGES), sorted."""
    found = set()
    for path in paths:
        found.update(CJK_PATTERN.findall(read(path)))
    return "".join(sorted(found))


def main(argv=None, docs=DOCS, languages_dir=LANGUAGES_DIR):
    parser = argparse.ArgumentParser(description="Keep the translated docs pages in step with the English ones.")
    parser.add_argument("command", nargs="?", choices=("check", "show", "stamp", "generate"), default="check")
    parser.add_argument("keys", nargs="*", help="section ids for show and stamp")
    parser.add_argument("--page", help="only this page, such as quickstart")
    parser.add_argument("--language", help="only this language, such as zh_CN")
    parser.add_argument("--all", action="store_true", help="stamp: every section listed as stale or unstamped")
    parser.add_argument("--check", action="store_true", help="generate: change nothing; exit 1 if the generated parts are out of date")
    args = parser.parse_args(argv)

    if args.command == "generate":
        stale = generate(docs, languages_dir, check=args.check)
        for path, what in stale:
            print("%s: %s" % (_display(path), what))
        if args.check:
            if stale:
                print("\nRun `python src/resources/languages/Docs_Translation.py generate` to write them.", file=sys.stderr)
                return 1
            print("The language dropdowns and language.css files are as the language folders give them.")
            return 0
        problems = [item for item in stale if "no %s" % LANGUAGES_BEGIN in item[1] or "can't be used" in item[1]]
        print("Wrote %d file(s)." % (len(stale) - len(problems)) if stale else "Nothing to write: all is as the language folders give it.")
        return 1 if problems else 0

    pairs = [pair for pair in translations(docs, languages_dir)
             if args.page in (None, pair[0]) and args.language in (None, pair[1])]
    if not pairs:
        what = " and ".join(part for part in (
            "page %s" % args.page if args.page else "", "language %s" % args.language if args.language else "") if part)
        print("No translated page%s." % (" for " + what if what else ""), file=sys.stderr)
        return 2 if what else 0

    if args.command == "show":
        found = 0
        for page, code, english_path, translated_path in pairs:
            english, translated = Page(english_path), Page(translated_path)
            if not all(english.unit(key) is not None for key in args.keys):
                continue
            found += 1
            for label, shown, where in (("English", english, english_path), ("Translation", translated, translated_path)):
                for key in args.keys:
                    source = section_source(shown, key)
                    print("===== %s: %s (%s) =====" % (label, key, _display(where)))
                    print(source if source is not None else "(no such section)")
        if not found:
            print("No such section on a page: %s" % ", ".join(args.keys), file=sys.stderr)
            return 2
        return 0

    if args.command == "stamp":
        chosen = pairs
        if args.keys:
            chosen = [pair for pair in pairs if all(Page(pair[2]).unit(key) is not None for key in args.keys)]
            if not chosen:
                print("No such section on a page: %s" % ", ".join(args.keys), file=sys.stderr)
                return 2
            if len(chosen) > 1:
                print(
                    "These sections are on more than one page or language (%s); name one with --page and --language, "
                    "because a stamp says that translation was reviewed." % ", ".join(
                        "%s/%s" % (pair[0], pair[1]) for pair in chosen), file=sys.stderr)
                return 2
        stamped = []
        for page, code, english_path, translated_path in chosen:
            report = check_page(english_path, translated_path, code)
            if report.problems:
                print(report.format())
                print("\nNothing was stamped: first make the pages match, as listed above.", file=sys.stderr)
                return 1
            keys = list(args.keys)
            if args.all:
                keys += report.stale + report.unstamped
            if keys:
                done = stamp_sections(english_path, translated_path, keys)
                stamped.append("%s/%s: %s" % (page, code, ", ".join(done)))
        if not stamped:
            reports = check_all(docs, languages_dir, args.page, args.language)
            left = ["%s/%s: %s" % (r.page_name, r.language, ", ".join(r.stale + r.unstamped)) for r in reports if r.stale + r.unstamped]
            print("Name the sections to stamp, or use --all. Stale: %s." % ("; ".join(left) or "none"))
            return 1
        print("Stamped %s." % "; ".join(stamped))
        return 0

    reports = [check_page(english, translated, code) for page, code, english, translated in pairs]
    failed = False
    for report in reports:
        if report.ok:
            print("%s matches %s." % (report.translated, report.english))
        else:
            failed = True
            print(report.format())
    for message in unused_rules(docs):
        failed = True
        print(message)
    return 1 if failed else 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
