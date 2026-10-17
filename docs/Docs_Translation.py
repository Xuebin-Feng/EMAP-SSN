# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Keeps a translated documentation page in step with its English page.

A translation of docs/<page>.html lives at docs/<language>/<page>.html, such as
docs/zh_CN/list_of_commands.html. It is the English page with the readable text
translated and nothing else changed: the same ids, links, <code> snippets,
styles and script logic. This tool checks that, and tells which sections of the
English page changed after the translation was last reviewed.

Each translated section carries data-source-sha256, the fingerprint of its
English section as it was when the translation was reviewed. The fingerprint is
taken over the section's prose and code only: its text with whitespace
collapsed, the text of <code> and command lines, and the translatable
attributes. Tags, classes and styles are not in it, so restyling the English
page makes no translation stale (the two pages' tags are compared with each
other instead, below). After the English page's text changes, `check` names the
sections whose fingerprint no longer matches; bring each up to date in the
translated page, then `stamp` it. A section is each element with an id that
sits beside the page's <section>s (a card, a divider, the header), each other
child of <body> outside the block around them (the sidebar, the inline script),
and the <title>.

    python docs/Docs_Translation.py                   # what needs an update
    python docs/Docs_Translation.py show cmd-label    # the English and translated section
    python docs/Docs_Translation.py stamp cmd-label   # after translating that section
    python docs/Docs_Translation.py stamp --all       # every section listed as stale
    python docs/Docs_Translation.py characters --out chars.txt   # the font's character list

Options: --page NAME (default list_of_commands) and --language CODE (default
zh_CN, the folder under docs/). The tool uses only the standard library and
never changes the English page. `check` exits with 1 while anything is left.
`characters` writes the Chinese characters the pages draw from the Noto Sans SC
subsets in docs/fonts (see docs/fonts/LICENSE.fonts for rebuilding them).

What `check` compares, apart from the fingerprints:
  - the same sections in the same order, and the same tags, ids, classes, event
    handlers and links around and inside them (a link to a file is compared by
    the file it names, so ../quickstart.html matches quickstart.html);
  - the same text in every <code>, in the command names and syntax boxes, in
    the example lines and in the sidebar's command list (VERBATIM_RULES): what
    readers type is never translated;
  - the same <style> or the same stylesheet link (the two pages are compared
    with each other, never with fixed contents), and the same script apart from
    the strings shown to readers (TRANSLATED_SCRIPT_STRINGS), each of which
    must be translated; the theme and sidebar code is the shared assets/docs.js,
    which shows the labels its button carries (TRANSLATED_ATTRIBUTES), and those
    too must be translated;
  - the language switch: the current language is marked, the other one links
    to the sibling page, and that file exists.
The translatable attributes (title, placeholder, aria-label) must be present
but may differ.
"""

import argparse
import hashlib
import html.parser
import json
import posixpath
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
DEFAULT_PAGE = "list_of_commands"
DEFAULT_LANGUAGE = "zh_CN"
STAMP = "data-source-sha256"

# Strings in the page's script that readers see: they differ between the pages,
# and the translated page must not keep the English one. Any other string in the
# script is code and must be the same.
TRANSLATED_SCRIPT_STRINGS = ("Copied!",)

# Attributes whose text docs/assets/docs.js shows to readers: the theme button's label for
# each mode, which the shared script reads from the page, so each page writes it in its
# own language. They differ between the pages, and the translated page must not keep the
# English text.
TRANSLATED_ATTRIBUTES = ("data-label-dark", "data-label-light")

# Attributes whose text is for readers: present on both pages, free to differ.
TRANSLATABLE_ATTRIBUTES = ("title", "placeholder", "aria-label", "alt") + TRANSLATED_ATTRIBUTES

# The characters that docs/fonts/fonts_zh_CN.css takes from the Noto Sans SC subsets: its
# unicode-range. Everything else on a page comes from the Latin faces or the system.
CJK_RANGES = (
    (0x2E80, 0x2FDF), (0x3000, 0x303F), (0x3040, 0x30FF), (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFE30, 0xFE4F), (0xFF00, 0xFFEF),
)
CJK_PATTERN = re.compile("[" + "".join("%s-%s" % (chr(low), chr(high)) for low, high in CJK_RANGES) + "]")

VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
}

USAGE = (
    "python docs/Docs_Translation.py              lists what needs an update\n"
    "python docs/Docs_Translation.py show KEY     prints the English and the translated section\n"
    "python docs/Docs_Translation.py stamp KEY... after reviewing those sections (or: stamp --all)"
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
    """A parsed page and its sections."""

    def __init__(self, path):
        self.path = Path(path)
        self.source = read(path)
        self.root = parse(self.source)
        self.html = first(self.root, "html")
        self.head = first(self.root, "head")
        self.body = first(self.root, "body")
        self.problems = []
        self.units = self._find_units()

    def _find_units(self):
        """(key, element) for the page's sections, in document order.

        The <title>; every child with an id of the block that holds the <section>s;
        and every other child of <body> or of the blocks around that one (the
        sidebar, the inline script), named by its id or its tag."""
        units = []
        if self.head is None or self.body is None or self.html is None:
            self.problems.append("%s has no <html>, <head> and <body>." % self.path.name)
            return units
        title = first(self.head, "title")
        if title is not None:
            units.append(("title", title))
        sections = [e for e in self.body.walk() if e.tag == "section"]
        if not sections:
            self.problems.append("%s has no <section> to find the page's content by." % self.path.name)
            return units
        container = sections[0].parent
        around = set()
        node = container.parent
        while node is not None:
            around.add(id(node))
            node = node.parent
        used = {}

        def add(key, element):
            used[key] = used.get(key, 0) + 1
            units.append((key if used[key] == 1 else "%s-%d" % (key, used[key]), element))

        def descend(node):
            for child in node.elements():
                if child is container:
                    for member in child.elements():
                        key = member.attrs.get("id")
                        if key:
                            add(key, member)
                        else:
                            self.problems.append(
                                "%s: a <%s> beside the <section>s has no id (%s); give every such element an id so its "
                                "translation can be matched." % (self.path.name, member.tag, collapse(member.start_tag)[:70])
                            )
                elif id(child) in around:
                    descend(child)
                elif child.tag == "script" and "src" in child.attrs:
                    continue
                else:
                    add(child.attrs.get("id") or child.tag, child)

        descend(self.body)
        keys = [key for key, _ in units]
        for key in sorted({key for key in keys if keys.count(key) > 1}):
            self.problems.append("%s: the section id '%s' is used more than once." % (self.path.name, key))
        return units

    def unit(self, key):
        for name, element in self.units:
            if name == key:
                return element
        return None

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


def fingerprint(element):
    data = json.dumps(canonical(element), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---- Structure ----------------------------------------------------------


def _link_target(page_path, value):
    """A link, compared by what it names: a fragment or a URL as written, a file by its resolved path."""
    if not value or value.startswith("#") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) or value.startswith("//"):
        return value
    target, _, fragment = value.partition("#")
    resolved = posixpath.normpath(posixpath.join(Path(page_path).parent.as_posix(), target))
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


def _skeleton(page, element):
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

    walk(element)
    return tokens, verbatim


def _frame(page):
    """The page outside its sections: head, body, the blocks around them and the styles, with each section as a marker."""
    section_of = {id(element): key for key, element in page.units}
    tokens = []

    def walk(node):
        key = section_of.get(id(node))
        if key is not None:
            tokens.append(("section", key))
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
                "add it to TRANSLATED_SCRIPT_STRINGS in docs/Docs_Translation.py; otherwise copy it unchanged."
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
    english_tokens, english_verbatim = _skeleton(english_page, english)
    translated_tokens, translated_verbatim = _skeleton(translated_page, translated)
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
    english_frame, translated_frame = _frame(english_page), _frame(translated_page)
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


# ---- The language switch ------------------------------------------------


def language_attribute(code):
    return code.replace("_", "-")


def _language_switch_problems(page, language, siblings):
    """The switch on one page: the current language marked, a working link to the other page."""
    problems = []
    switch = next((e for e in page.root.walk() if "lang-switch" in e.classes), None)
    if switch is None:
        return ['%s has no language switch (a <nav class="lang-switch"> beside the theme button).' % page.path.name]
    options = [e for e in switch.elements() if "lang-option" in e.classes]
    expected = [lang for lang, _ in siblings]
    if [e.attrs.get("lang") for e in options] != expected:
        return [
            "%s: the language switch should list %s in this order, by their lang attributes; it has %s."
            % (page.path.name, expected, [e.attrs.get("lang") for e in options])
        ]
    for option, (lang, target) in zip(options, siblings):
        if lang == language:
            if option.tag == "a" or "current" not in option.classes or option.attrs.get("aria-current") != "page":
                problems.append(
                    "%s: the switch must mark %s, the page's own language, as current (a <span> with the classes "
                    "'lang-option current' and aria-current=\"page\")." % (page.path.name, lang)
                )
            continue
        if option.tag != "a" or "current" in option.classes:
            problems.append("%s: the switch's entry for %s must be a link that is not marked current." % (page.path.name, lang))
            continue
        base = option.attrs.get("href", "").split("#")[0]
        resolved = (page.path.parent / base).resolve() if base else None
        if resolved is None or resolved != Path(target).resolve():
            problems.append("%s: the switch's link for %s points at '%s'; it should reach %s." % (page.path.name, lang, base, target))
        elif not resolved.is_file():
            problems.append("%s: the switch's link for %s points at %s, which does not exist." % (page.path.name, lang, resolved))
    return problems


# ---- The check ----------------------------------------------------------


class Report:
    def __init__(self, language, page_name):
        self.language = language
        self.page_name = page_name
        self.problems = []  # the pages differ in more than text: fix before anything else
        self.stale = []  # sections whose English changed since the translation was reviewed
        self.unstamped = []  # sections without a fingerprint

    @property
    def ok(self):
        return not (self.problems or self.stale or self.unstamped)

    def format(self):
        lines = []
        english = "docs/%s.html" % self.page_name
        translated = "docs/%s/%s.html" % (self.language, self.page_name)
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
            lines.append("How to bring the translation up to date:")
            lines.extend("  " + line for line in USAGE.splitlines())
        return "\n".join(lines)


def check_page(english_path, translated_path, language=DEFAULT_LANGUAGE):
    """Compares a translated page with its English page; returns a Report."""
    english_path, translated_path = Path(english_path), Path(translated_path)
    report = Report(language, english_path.stem)
    english, translated = Page(english_path), Page(translated_path)
    report.problems.extend(english.problems)
    report.problems.extend(translated.problems)

    for (description, _), count in zip(VERBATIM_RULES, english.verbatim_rule_counts()):
        if count == 0:
            report.problems.append(
                "VERBATIM_RULES in docs/Docs_Translation.py has the rule '%s', which matches nothing on the English page "
                "any more. Update the rule to the page's new markup, so the text it protects is still checked." % description
            )

    script_strings = set()
    for key, element in english.units:
        if element.tag == "script":
            script_strings.update(_script_parts(_script_text(element))[1])
    for text in TRANSLATED_SCRIPT_STRINGS:
        if text not in script_strings:
            report.problems.append(
                "TRANSLATED_SCRIPT_STRINGS in docs/Docs_Translation.py lists '%s', which the English page's script no "
                "longer has. List the strings the script shows to readers now." % text
            )

    for name in TRANSLATED_ATTRIBUTES:
        english_values = [e.attrs[name] for e in english.root.walk() if name in e.attrs]
        translated_values = [e.attrs[name] for e in translated.root.walk() if name in e.attrs]
        if not english_values:
            report.problems.append(
                "TRANSLATED_ATTRIBUTES in docs/Docs_Translation.py lists '%s', which no element of the English page has "
                "any more. List the attributes the shared script shows to readers now." % name
            )
        elif len(english_values) == len(translated_values):
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

    switches = [("en", english_path), (expected_lang, translated_path)]
    report.problems.extend(_language_switch_problems(english, "en", switches))
    report.problems.extend(_language_switch_problems(translated, expected_lang, switches))

    for key in shared:
        stamp = translated.unit(key).attrs.get(STAMP)
        if not stamp:
            report.unstamped.append(key)
        elif stamp != fingerprint(english.unit(key)):
            report.stale.append(key)
    return report


def english_fingerprints(english_path):
    return {key: fingerprint(element) for key, element in Page(english_path).units}


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
        value = '%s="%s"' % (STAMP, fingerprint(english.unit(key)))
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
    element = page.unit(key)
    return page.source[element.start : element.end] if element is not None else None


# ---- Command line -------------------------------------------------------


def cjk_characters(*paths):
    """The characters of the pages that fonts_zh_CN.css draws from the Noto Sans SC subsets, sorted."""
    found = set()
    for path in paths:
        found.update(CJK_PATTERN.findall(read(path)))
    return "".join(sorted(found))


def main(argv=None, docs=DOCS):
    parser = argparse.ArgumentParser(description="Keep a translated docs page in step with its English page.")
    parser.add_argument("command", nargs="?", choices=("check", "show", "stamp", "characters"), default="check")
    parser.add_argument("keys", nargs="*", help="section ids for show and stamp")
    parser.add_argument("--page", default=DEFAULT_PAGE)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--all", action="store_true", help="stamp: every section listed as stale or unstamped")
    parser.add_argument("--out", help="characters: write the list to this file (UTF-8) instead of printing it")
    args = parser.parse_args(argv)

    english_path = Path(docs) / (args.page + ".html")
    translated_path = Path(docs) / args.language / (args.page + ".html")
    for path in (english_path, translated_path):
        if not path.is_file():
            print("No such page: %s" % path, file=sys.stderr)
            return 2

    if args.command == "characters":
        characters = cjk_characters(english_path, translated_path)
        if args.out:
            Path(args.out).write_bytes(characters.encode("utf-8"))
            print("%d characters written to %s." % (len(characters), args.out))
        else:
            print(characters)
        return 0

    if args.command == "show":
        english, translated = Page(english_path), Page(translated_path)
        for key in args.keys:
            for label, page in (("English", english), ("Translation", translated)):
                source = section_source(page, key)
                print("===== %s: %s =====" % (label, key))
                print(source if source is not None else "(no such section)")
        return 0

    report = check_page(english_path, translated_path, args.language)
    if args.command == "stamp":
        if report.problems:
            print(report.format())
            print("\nNothing was stamped: first make the pages match, as listed above.", file=sys.stderr)
            return 1
        keys = list(args.keys)
        if args.all:
            keys += report.stale + report.unstamped
        if not keys:
            print("Name the sections to stamp, or use --all. Stale: %s." % (", ".join(report.stale + report.unstamped) or "none"))
            return 1
        done = stamp_sections(english_path, translated_path, keys)
        print("Stamped %d section(s): %s." % (len(done), ", ".join(done)))
        return 0

    if report.ok:
        print("docs/%s/%s.html matches docs/%s.html." % (args.language, args.page, args.page))
        return 0
    print(report.format())
    return 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
