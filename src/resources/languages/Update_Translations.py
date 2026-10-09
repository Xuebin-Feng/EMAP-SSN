# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Update EMAP-SSN's translation catalogs from the code.

    python src/resources/languages/Update_Translations.py            update and compile
    python src/resources/languages/Update_Translations.py --add de   start a German catalog
    python src/resources/languages/Update_Translations.py --check    change nothing; fail if stale

The update collects every text marked for translation in src:

* self.tr("..."), QCoreApplication.translate("Context", "...") and
  QT_TRANSLATE_NOOP("Context", "..."), which Qt's lupdate finds;
* the English template of every Message("...", ...), the Viewer's console
  line, filed under the "Message" context. lupdate can't read a Message, so
  in a temporary copy of the code each one gets a QT_TRANSLATE_NOOP on the
  same line, and translators still see the real file and line;
* the texts of the web pages the Viewer serves (web_ui/Page_Texts.py tells
  how a page marks them), filed under each page's context. The temporary
  copy of a page holds only a QT_TRANSLATE_NOOP per text, on its line.

A counted text, such as translate("Config", "%n file(s)", None, count),
needs a plural form per language. lupdate sees the count of translate()
only when it is a number as written, and drops the text when it is a call
such as len(files), so the temporary copy writes 0 in its place. lupdate
also drops or miscounts a text whose call names its arguments, keeps the
new line where a backslash ends a line inside a text, and a counted tr()
would show English as "1 file(s)", so the update refuses those.

It rewrites emapssn.ts, the list of every text, and merges the texts into
each language's catalog, emapssn_<language>.ts, keeping its translations. A
text that left the code stays there, marked obsolete, in case it comes back.
Then it compiles each language's catalog into emapssn_<language>.qm, the
file the program loads, and checks that every translation keeps its
text's placeholders, such as {count}.

It also lists the translated tool help pages, <name>.<language>.md beside
src/tools/tool_descriptions/<name>.md, whose English page changed since:
Tools shows those in English until they are brought up to date
(utilities/Help_Pages.py). That is a note, not a failure.

--check changes nothing. It exits with 1 if emapssn.ts no longer lists the
code's texts, if a compiled catalog no longer matches its .ts file, or if a
translation lost a placeholder. The test suite runs it, so a text marked
without running the update fails the tests.

update_catalogs also keeps a window's own catalog beside the main one:
opt_vr's Update_Translations_VR.py keeps VR Config's emapssn_vr.ts from
opt_vr/src, leaving out every text emapssn.ts already lists, which VR
Config shows from the main catalog.
"""

import argparse
import ast
import io
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize

LANGUAGES_DIR = Path(__file__).resolve().parent
SRC_DIR = LANGUAGES_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities.Localization import (  # noqa: E402
    CATALOG_NAME,
    MESSAGE_CONTEXT,
    fill_ins,
    has_plural_ending,
    read_catalog,
)

LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(?:_[A-Z][a-z]{3})?(?:_[A-Z]{2})?$")
_MARKER_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}
# lupdate reads .py files as Python, .js as JavaScript and any other, such
# as a staged page's .html, as C++; each reads a QT_TRANSLATE_NOOP.
_SCANNED_EXTENSIONS = "py,html,js"
# lupdate's own progress lines; anything else it prints is worth showing.
_LUPDATE_PROGRESS = re.compile(
    r"^\s*(Scanning directory|Updating|Found \d+ source text|Removed \d+ obsolete)"
)


def qt_tool(name):
    """The path of one of Qt's translation tools, as PySide6 ships it."""
    import PySide6

    folder = Path(PySide6.__file__).resolve().parent
    for candidate in (folder / f"{name}.exe", folder / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Qt's {name} is not in {folder}. Reinstall PySide6.")


def message_marker(template, counted=False):
    """The Message template as lupdate reads it, on one line of Python, then ", ".

    QT_TRANSLATE_NOOP("Message", template), or for a counted template,
    whose plural forms need a count, translate("Message", template, None, 0).
    """
    literal = _marker_literal(template)
    if counted:
        return f'QCoreApplication.translate("{MESSAGE_CONTEXT}", "{literal}", None, 0), '
    return f'QT_TRANSLATE_NOOP("{MESSAGE_CONTEXT}", "{literal}"), '


def _marker_literal(text):
    return "".join(_MARKER_ESCAPES.get(character, character) for character in text)


def page_marker(context, text):
    """A page's text as lupdate reads it in the page's staged copy, then a space."""
    return f'QT_TRANSLATE_NOOP("{context}", "{_marker_literal(text)}"); '


# The argument that holds the text, for each way of marking one, and the
# argument that holds a counted text's count.
_TEXT_ARGUMENT = {"tr": 0, "QT_TR_NOOP": 0, "translate": 1, "QT_TRANSLATE_NOOP": 1}
_COUNT_ARGUMENT = {"tr": 2, "translate": 3}


def _called_name(node):
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _is_plain_string(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _is_filled_in_string(node):
    """An f-string, or a string filled in with % or .format() or joined with +."""
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
        return any(_is_plain_string(side) or _is_filled_in_string(side) for side in (node.left, node.right))
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format" and _is_plain_string(node.func.value)
    )


def _is_count_as_written(node):
    return isinstance(node, ast.Constant) and type(node.value) is int


# An odd run of backslashes before a newline: the last one joins the lines.
_LINE_JOIN = re.compile(r"(?<!\\)(?:\\\\)*\\\n")


def _joins_lines(text, node):
    """Whether a string literal of node ends a line with a backslash.

    Python joins the lines, but lupdate keeps the newline, so the catalog
    would list a text the code never shows.
    """
    segment = ast.get_source_segment(text, node) or ""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(segment).readline))
    except (tokenize.TokenError, SyntaxError):
        return False
    return any(token.type == tokenize.STRING and _LINE_JOIN.search(token.string) for token in tokens)


_LINE_JOIN_ADVICE = ("ends a line with a backslash inside the text, which lupdate reads as a new "
                     "line. Write the text without it, or join plain literals.")


def _column(line, offset):
    """The character column of ast's offset, which counts UTF-8 bytes, in line."""
    return len(line.encode("utf-8")[:offset].decode("utf-8"))


_PLURAL_ADVICE = ('Mark its English plural ending, as in "%n file(s)" or "%n match(es)", '
                  'so English shows "1 file" and "2 files".')


def prepare_source(text, filename, shared=frozenset()):
    """lupdate's view of one source file, and the problems in its marked texts.

    Returns the text as lupdate should read it, and a list of problems. In
    that text, each Message(...) call has a message_marker right after its
    "(", and the count of each counted translate() is 0, which lupdate
    sees as a count where it would miss a call such as len(files). The
    problems are texts no catalog can list or lupdate would get wrong: a
    Message whose template isn't a plain string, a text filled in before
    tr() or translate() gets it, a call that names its arguments, and a
    text with %n that is not counted, or counted without an English plural
    ending. The changes add no lines, so every text keeps its line number.

    shared holds the keys (context, text, disambiguation) of texts another
    catalog lists. lupdate doesn't see those: a Message gets no marker, and
    translate() and QT_TRANSLATE_NOOP() are renamed so lupdate skips them.
    """
    try:
        tree = ast.parse(text, filename)
    except SyntaxError as error:
        return text, [
            f"{filename}:{error.lineno}: not valid Python, so its texts were not collected: {error.msg}"
        ]
    lines = text.splitlines(keepends=True)
    edits, problems = [], []  # Each edit: row, column, end row, end column, new text.

    def problem(node, explanation):
        problems.append((node.lineno, f"{filename}:{node.lineno}: {explanation}"))

    def check_message(node):
        template = node.args[0] if node.args else None
        if not _is_plain_string(template):
            problem(node, "Message needs its English template as a plain string, not an f-string "
                          "or a variable, so the catalog can list it.")
            return
        if _joins_lines(text, template):
            problem(node, f"Message's template {_LINE_JOIN_ADVICE}")
        counted = "%n" in template.value
        if counted and not {keyword.arg for keyword in node.keywords} & {"n", None}:
            problem(node, "Message's template holds %n, Qt's count, so it needs the count as n=, "
                          'as in Message("Removed %n group(s).", n=count).')
        if counted and not has_plural_ending(template.value):
            problem(node, f"Message counts a template without an English plural ending. {_PLURAL_ADVICE}")
        if (MESSAGE_CONTEXT, template.value, "") in shared:
            return
        row = node.func.end_lineno - 1
        line = lines[row]
        start = _column(line, node.func.end_col_offset)
        column = line.find("(", start)
        if column < 0 or line[start:column].strip():
            problem(node, "put Message's opening parenthesis on the same line as its name.")
            return
        edits.append((row, column + 1, row, column + 1, message_marker(template.value, counted)))

    def shared_key(node, name, source):
        """The key of a translate() or QT_TRANSLATE_NOOP() text, if shared lists it."""
        if name not in ("translate", "QT_TRANSLATE_NOOP") or not _is_plain_string(node.args[0]):
            return None
        disambiguation = node.args[2] if name == "translate" and len(node.args) > 2 else None
        comment = disambiguation.value if disambiguation is not None and _is_plain_string(disambiguation) else ""
        key = (node.args[0].value, source.value, comment)
        return key if key in shared else None

    def leave_out(node, name):
        """Rename the call in lupdate's copy, so lupdate skips it."""
        row = node.func.end_lineno - 1
        end = _column(lines[row], node.func.end_col_offset)
        edits.append((row, end - len(name), row, end, f"_shared_{name}"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node)
        if name == "Message":
            check_message(node)
            continue
        position = _TEXT_ARGUMENT.get(name)
        if position is None or len(node.args) <= position:
            continue
        source = node.args[position]
        if _is_filled_in_string(source):
            problem(node, f"{name}() gets a text that is already filled in, which no catalog can "
                          "list. Pass the plain text and fill in the values after translating.")
            continue
        if not _is_plain_string(source):
            continue  # A text chosen at run time, such as translate("Shapes", shape), or str.translate().
        if node.keywords:
            problem(node, f"{name}() names its arguments, so lupdate would drop the text or miss "
                          'its count. Pass them in order, as in self.tr("%n file(s)", "", count).')
            continue
        if _joins_lines(text, source):
            problem(node, f"{name}() gets a text that {_LINE_JOIN_ADVICE}")
        if shared and shared_key(node, name, source):
            leave_out(node, name)
            continue
        count = _COUNT_ARGUMENT.get(name)
        counted = count is not None and len(node.args) > count
        if counted and name == "tr":
            problem(node, "tr() counts a text, but only Desktop_App's translate() shows its English "
                          'plural. Write translate("Context", "%n file(s)", None, count).')
            continue
        if "%n" in source.value and not counted:
            problem(node, f"{name}() gets a text with %n, Qt's count, but no count, so %n would "
                          'show as written. Pass the count, as in self.tr("%n file(s)", "", count).')
        if counted and not has_plural_ending(source.value):
            problem(node, f"{name}() counts a text without an English plural ending. {_PLURAL_ADVICE}")
        if counted and name == "translate" and not _is_count_as_written(node.args[count]):
            argument = node.args[count]
            row, end_row = argument.lineno - 1, argument.end_lineno - 1
            edits.append((
                row, _column(lines[row], argument.col_offset),
                end_row, _column(lines[end_row], argument.end_col_offset), "0",
            ))
    for row, column, end_row, end_column, new in sorted(edits, reverse=True):
        # Newlines in place of the lines replaced keep every later line's number.
        lines[row] = lines[row][:column] + new + "\n" * (end_row - row) + lines[end_row][end_column:]
        del lines[row + 1:end_row + 1]
    return "".join(lines), [explanation for _, explanation in sorted(problems)]


def stage_sources(source_dir, stage_dir, skip=None, shared=frozenset()):
    """Copy the .py files under source_dir into stage_dir, marking Message templates.

    Files in the folder skip (relative to source_dir), such as this command's
    own, are left out, and so are the texts shared lists, as prepare_source
    explains. Returns the problems found.
    """
    source_dir, stage_dir = Path(source_dir), Path(stage_dir)
    skipped = Path(skip).parts if skip else None
    problems = []
    for path in sorted(source_dir.rglob("*.py")):
        relative = path.relative_to(source_dir)
        if relative.parts[: len(skipped or ())] == skipped or any(
            part == "__pycache__" or part.startswith(".") for part in relative.parts
        ):
            continue
        text = path.read_text(encoding="utf-8-sig")
        if any(word in text for word in ("Message", "tr(", "translate(", "QT_TR")):
            text, found = prepare_source(text, relative.as_posix(), shared)
            problems.extend(found)
        target = stage_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    return problems


def stage_pages(source_dir, stage_dir):
    """Write lupdate's view of each web page under source_dir into stage_dir.

    lupdate can't read a page's markup or its t() calls, so the staged copy
    of a page (web_ui/Page_Texts.py lists them) holds only a page_marker
    per marked text, on the text's own line. Returns the problems found.
    """
    from web_ui.Page_Texts import PAGE_CONTEXTS, read_page

    problems = []
    for relative, context in PAGE_CONTEXTS.items():
        path = Path(source_dir) / relative
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8-sig")
        page = read_page(source, path.suffix)
        problems += [f"{relative}:{line}: {explanation}" for line, explanation in page.problems]
        lines = [""] * (source.count("\n") + 1)
        for text in page.texts:
            if "\\" in text.text and path.suffix == ".js":
                # lupdate reads a script's escapes twice, so "\\" comes out as nothing.
                problems.append(f"{relative}:{text.line}: {text.text!r} holds a backslash, which lupdate "
                                "misreads in a script. Write the text without one.")
                continue
            lines[text.line - 1] += page_marker(context, text.text)
        target = Path(stage_dir) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return problems


def help_page_reports(source_dir):
    """A line for each translated tool help page under source_dir that Tools shows in English.

    Help_Pages.help_page_text shows a translation only while its first line
    matches its English page, so these need bringing up to date. They are
    reported, not refused: the window shows the English page meanwhile.
    """
    from utilities import Help_Pages

    lines = []
    for translation, page in Help_Pages.stale_translations(Path(source_dir) / "tools" / "tool_descriptions"):
        if not page.is_file():
            lines.append(f"{translation.name} translates {page.name}, which no longer exists, so Tools never shows it.")
        else:
            lines.append(
                f"{translation.name} was translated from another version of {page.name}, so Tools shows the "
                f"English page. Bring it up to date, then make its first line:\n  {Help_Pages.translation_marker(page)}"
            )
    return lines


def run_qt_tool(name, arguments):
    """Run lupdate or lrelease. Returns what it printed beyond its progress lines."""
    result = subprocess.run(
        [str(qt_tool(name)), *map(str, arguments)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    printed = [
        line for line in (result.stdout + result.stderr).splitlines()
        if line.strip() and not _LUPDATE_PROGRESS.match(line)
    ]
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}:\n" + "\n".join(printed))
    return printed


def language_catalogs(languages_dir, catalog_name=CATALOG_NAME):
    """{language: path} of every <catalog_name>_<language>.ts in languages_dir."""
    pattern = re.compile(rf"^{re.escape(catalog_name)}_(?P<language>\w+)\.ts$")
    catalogs = {}
    for path in sorted(Path(languages_dir).glob(f"{catalog_name}_*.ts")):
        match = pattern.match(path.name)
        if match and LANGUAGE_CODE.match(match.group("language")):
            catalogs[match.group("language")] = path
    return catalogs


def placeholder_problems(catalog):
    """Translations in catalog that dropped or added a placeholder.

    A plural form may leave out %n, as in "one file". Texts that left the
    code are not compiled, so they are not checked.
    """
    problems = []
    for message in filter(_is_live, read_catalog(catalog)):
        counts = ("%n", "%Ln") if message.numerus else ()
        expected = [fill_in for fill_in in fill_ins(message.source) if fill_in not in counts]
        for translation in message.translations:
            found = [fill_in for fill_in in fill_ins(translation) if fill_in not in counts]
            if translation and found != expected:
                problems.append(
                    f"{Path(catalog).name}: [{message.context}] {message.source!r} is translated as "
                    f"{translation!r}, whose placeholders {found} differ from {expected}."
                )
    return problems


def _texts(catalog):
    return {
        (message.context, message.source, message.comment, message.numerus)
        for message in read_catalog(catalog)
    }


def _same_file(first, second):
    first, second = Path(first), Path(second)
    return first.is_file() and second.is_file() and first.read_bytes() == second.read_bytes()


def _is_live(message):
    """Whether a catalog's message is still in the code; lrelease leaves the others out."""
    return message.status not in ("vanished", "obsolete")


def update_catalogs(
    source_dir=SRC_DIR, languages_dir=LANGUAGES_DIR, add=(), check=False, report=print,
    catalog_name=CATALOG_NAME, shared_template=None,
):
    """Bring the catalogs in languages_dir up to date with the code in source_dir.

    add starts catalogs for new languages. With check, nothing changes.
    catalog_name names the template, <catalog_name>.ts, and the languages'
    catalogs; shared_template is another template whose texts are left out.
    Returns 0, or 1 when something needs fixing; report gets every line to show.
    """
    source_dir, languages_dir = Path(source_dir).resolve(), Path(languages_dir).resolve()
    template = languages_dir / f"{catalog_name}.ts"
    languages = language_catalogs(languages_dir, catalog_name)
    shared = frozenset(message.key for message in read_catalog(shared_template)) if shared_template else frozenset()
    problems = []
    for language in add:
        if not LANGUAGE_CODE.match(language):
            raise ValueError(f"{language!r} is not a language code such as de, pt_BR or zh_CN.")
        languages.setdefault(language, languages_dir / f"{catalog_name}_{language}.ts")

    with tempfile.TemporaryDirectory(prefix="emapssn-catalogs-") as temporary:
        # lupdate records each text's file relative to its catalog, so the
        # copy keeps the catalogs where they are in the real tree.
        stage = Path(temporary) / source_dir.name
        stage_languages = stage / languages_dir.relative_to(source_dir)
        stage_languages.mkdir(parents=True)
        problems += stage_sources(source_dir, stage, skip=languages_dir.relative_to(source_dir), shared=shared)
        problems += stage_pages(source_dir, stage)

        fresh_template = stage_languages / template.name
        for line in run_qt_tool("lupdate", [
            "-extensions", _SCANNED_EXTENSIONS, "-source-language", "en", "-locations", "none", "-no-obsolete",
            stage, "-ts", fresh_template,
        ]):
            report(f"lupdate: {line}")
        texts = _texts(fresh_template)
        messages = sum(1 for context, *_ in texts if context == MESSAGE_CONTEXT)
        count = f"{len(texts)} texts, {messages} of them console messages"
        if check:
            listed = _texts(template) if template.is_file() else set()
            for context, source, _, _ in sorted(texts - listed):
                problems.append(f"{template.name} lacks [{context}] {source!r}: run the update.")
            for context, source, _, _ in sorted(listed - texts):
                problems.append(
                    f"{template.name} still lists [{context}] {source!r}, "
                    "which the code no longer has: run the update."
                )
            if texts == listed:
                report(f"{template.name} lists the code's {count}.")
        elif not _same_file(fresh_template, template):
            shutil.copyfile(fresh_template, template)
            report(f"Updated {template.name}: {count}.")
        else:
            report(f"{template.name} is up to date: {count}.")

        if languages and not check:
            staged = {}
            for language, catalog in languages.items():
                staged[language] = stage_languages / catalog.name
                if catalog.is_file():
                    shutil.copyfile(catalog, staged[language])
                else:
                    # lupdate would guess the language from the file name
                    # (de_DE for de); a new catalog names it exactly.
                    staged[language].write_text(
                        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE TS>\n'
                        f'<TS version="2.1" language="{language}" sourcelanguage="en">\n</TS>\n',
                        encoding="utf-8", newline="\n",
                    )
            for line in run_qt_tool("lupdate", [
                "-extensions", _SCANNED_EXTENSIONS, "-source-language", "en", "-locations", "relative",
                stage, "-ts", *staged.values(),
            ]):
                report(f"lupdate: {line}")
            for language, catalog in languages.items():
                if not _same_file(staged[language], catalog):
                    shutil.copyfile(staged[language], catalog)
                    report(f"Updated {catalog.name}.")

        for language, catalog in languages.items():
            if not catalog.is_file():
                continue
            problems += placeholder_problems(catalog)
            compiled = catalog.with_suffix(".qm")
            fresh = stage_languages / compiled.name
            run_qt_tool("lrelease", [catalog, "-qm", fresh])
            if not _same_file(fresh, compiled):
                if check:
                    problems.append(f"{compiled.name} does not match {catalog.name}: run the update.")
                else:
                    shutil.copyfile(fresh, compiled)
                    report(f"Compiled {compiled.name}.")
            live = list(filter(_is_live, read_catalog(catalog)))
            done = sum(1 for message in live if message.status == "" and any(message.translations))
            # Drafts are compiled and shown too, until a reviewer marks them finished.
            drafted = sum(1 for message in live if message.status == "unfinished" and any(message.translations))
            report(f"{catalog.name}: {done} of {len(live)} texts translated"
                   + (f"; {drafted} more drafted, awaiting review." if drafted else "."))

    for line in help_page_reports(source_dir):
        report(line)
    for problem in problems:
        report(problem)
    return 1 if problems else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Update EMAP-SSN's translation catalogs from the code, and compile them.",
    )
    parser.add_argument(
        "--check", action="store_true", help="change nothing; exit with 1 if a catalog is out of date",
    )
    parser.add_argument(
        "--add", metavar="LANGUAGE", action="append", default=[],
        help="start a catalog for LANGUAGE, such as de, pt_BR or zh_CN",
    )
    arguments = parser.parse_args(argv)
    if arguments.check and arguments.add:
        parser.error("--add changes the catalogs, so it can't be combined with --check")
    return update_catalogs(add=arguments.add, check=arguments.check)


if __name__ == "__main__":
    from utilities.Output_Streams import configure_output_streams

    configure_output_streams()  # Texts may hold any character; Windows pipes can't.
    sys.exit(main())
