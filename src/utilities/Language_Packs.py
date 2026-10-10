# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The languages EMAP-SSN shows: one folder each in src/resources/languages.

    src/resources/languages/<code>/
        emapssn_<code>.ts   the language's catalog, which a translator edits
        emapssn_<code>.qm   the compiled catalog, which the program loads
        language.json       optional: the language's name, font and punctuation
        help/<page>.md      optional: translated tool help pages (utilities/Help_Pages.py)

A folder is a language once its compiled catalog is in it, so adding a
language means adding its folder. The folder's name is the language's code,
such as de, pt_BR or zh_CN, and every name here is compared as the folder
lists it: zh_cn is not zh_CN, on Windows and macOS too, as on Linux.

language.json holds an object with any of three keys:

    "name"         the language's name in itself, as the Language dropdown
                   lists it. Without it, Qt's name for the code.
    "font"         a bundled font, for a script the core Noto faces lack:
                   "family" (its family name), "vispy_face" (the name the
                   Viewer's canvas registers it under), "files" (its regular
                   and bold faces, relative to src/resources/fonts/desktop) and
                   "web_range" (the CSS unicode-range the browser views draw in
                   it). Without it, the core faces draw the language.
    "punctuation"  how a message put together from sentences or list items is
                   punctuated (Localization.Punctuation): "no_space_after" (the
                   characters a sentence ends in without a space after it),
                   "separators" (a list's separators in the language's own
                   form, such as {", ": "，"}) and "separator_script" (the
                   unicode ranges whose text takes those forms). Without it,
                   the parts join as given.

Nothing here imports Qt, so the VR viewer process can use it too.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re

from utilities.Localization import CATALOG_NAME, LANGUAGES_DIR, Punctuation

LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(?:_[A-Z][a-z]{3})?(?:_[A-Z]{2})?$")
MANIFEST_NAME = "language.json"
HELP_FOLDER = "help"
_MANIFEST_KEYS = {"name", "font", "punctuation"}
_FONT_KEYS = {"family", "vispy_face", "files", "web_range"}
_PUNCTUATION_KEYS = {"no_space_after", "separators", "separator_script"}
_UNICODE_RANGE = re.compile(r"^[Uu]\+([0-9A-Fa-f]{1,6})(?:-([0-9A-Fa-f]{1,6}))?$")


class ManifestError(ValueError):
    """A language.json that can't be used, with the reason."""


@dataclass(frozen=True)
class LanguageFont:
    """A bundled font for a language whose script the core faces lack.

    Qt finds the family after the core family in each stack, so it shows
    only the characters the core lacks. VisPy draws a text in a single face,
    so the Viewer draws all its text in vispy_face, which has Latin letters
    too. files holds the regular and bold faces, relative to the desktop
    font folder. web_range is the CSS unicode-range of the script's
    characters, which the browser views draw in it.
    """

    family: str
    vispy_face: str
    files: tuple[str, ...]
    web_range: str = ""


@dataclass(frozen=True)
class LanguagePack:
    """A language's folder and what its language.json says; None where it says nothing."""

    code: str
    folder: Path
    name: str | None = None
    font: LanguageFont | None = None
    punctuation: Punctuation | None = None


# -- Names as the folders list them ---------------------------------------

def _listing(folder):
    """{name: is a directory} for the entries of folder, named exactly as it lists them."""
    try:
        with os.scandir(folder) as entries:
            return {entry.name: entry.is_dir() for entry in entries}
    except OSError:
        return {}


def exact_path(base, *names):
    """base/name/... if each folder on the way lists the next name exactly so, else None.

    Opening the path would also find another case on Windows and macOS,
    which Linux doesn't; this answers alike everywhere.
    """
    path = Path(base)
    for name in names:
        if name not in _listing(path):
            return None
        path = path / name
    return path


def exact_file(folder, name):
    """folder/name if folder lists a file of exactly that name, else None."""
    return Path(folder) / name if _listing(folder).get(name) is False else None


def language_folders(languages_dir=LANGUAGES_DIR):
    """The codes of every folder in languages_dir named as a language code, sorted."""
    return sorted(name for name, is_dir in _listing(languages_dir).items()
                  if is_dir and LANGUAGE_CODE.match(name))


def catalog_path(code, catalog_name=CATALOG_NAME, suffix=".ts", languages_dir=LANGUAGES_DIR):
    """Where a language's catalog lives: <languages_dir>/<code>/<catalog_name>_<code><suffix>."""
    return Path(languages_dir) / code / f"{catalog_name}_{code}{suffix}"


def language_codes(languages_dir=LANGUAGES_DIR, catalog_name=CATALOG_NAME, suffix=".qm"):
    """The languages in languages_dir whose catalog (compiled, by default) is in their folder, sorted."""
    return [code for code in language_folders(languages_dir)
            if exact_file(Path(languages_dir) / code, f"{catalog_name}_{code}{suffix}") is not None]


def find_catalog(code, catalog_name=CATALOG_NAME, suffix=".qm", languages_dir=LANGUAGES_DIR):
    """The language's catalog (compiled, by default), or None unless its folder and file are named exactly so.

    A name such as "../de" is in no folder's listing, so it finds nothing.
    """
    path = exact_path(languages_dir, str(code), f"{catalog_name}_{code}{suffix}")
    return path if path is not None and path.is_file() else None


def help_folder(code, languages_dir=LANGUAGES_DIR):
    """The folder of a language's translated tool help pages."""
    return Path(languages_dir) / code / HELP_FOLDER


# -- language.json --------------------------------------------------------

def unicode_ranges(text):
    """((first, last), ...) code points of a CSS unicode-range such as "U+3000-303F, U+FF00"."""
    ranges = []
    for part in str(text).split(","):
        match = _UNICODE_RANGE.match(part.strip())
        if match is None:
            raise ManifestError(f"{part.strip()!r} is not a range such as U+4E00-9FFF")
        first = int(match.group(1), 16)
        last = int(match.group(2), 16) if match.group(2) else first
        if last < first:
            raise ManifestError(f"{part.strip()!r} ends before it starts")
        ranges.append((first, last))
    return tuple(ranges)


def _text(value, what):
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{what} must be a non-empty string")
    return value


def _object(value, what, allowed):
    if not isinstance(value, dict):
        raise ManifestError(f"{what} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ManifestError(f"{what} has unknown keys {unknown}; it takes {sorted(allowed)}")
    return value


def _font(value):
    _object(value, '"font"', _FONT_KEYS)
    for key in ("family", "vispy_face", "files"):
        if key not in value:
            raise ManifestError(f'"font" needs "{key}"')
    files = value["files"]
    if not isinstance(files, list) or not files:
        raise ManifestError('"font" "files" must list the regular face, then the bold one')
    for path in files:
        _text(path, 'each of "font" "files"')
        written = PurePosixPath(path)
        if written.is_absolute() or ".." in written.parts or "\\" in path or ":" in path:
            raise ManifestError(f'"font" file {path!r} must be a path inside the desktop font folder, written with /')
    web_range = value.get("web_range", "")
    if web_range:
        unicode_ranges(web_range)
    return LanguageFont(_text(value["family"], '"font" "family"'), _text(value["vispy_face"], '"font" "vispy_face"'),
                        tuple(files), web_range)


def _punctuation(value):
    _object(value, '"punctuation"', _PUNCTUATION_KEYS)
    no_space_after = value.get("no_space_after", "")
    if not isinstance(no_space_after, str):
        raise ManifestError('"punctuation" "no_space_after" must be a string of characters')
    separators = value.get("separators", {})
    if not isinstance(separators, dict) or not all(
        isinstance(k, str) and k and isinstance(v, str) and v for k, v in separators.items()
    ):
        raise ManifestError('"punctuation" "separators" must map each separator to its form, such as {", ": "，"}')
    if " " in separators:
        raise ManifestError('"punctuation" "separators" can\'t change " ", which "no_space_after" governs')
    script = value.get("separator_script", "")
    if separators and not script:
        raise ManifestError('"punctuation" "separators" need a "separator_script", the ranges they apply next to')
    return Punctuation(frozenset(no_space_after), tuple(separators.items()), unicode_ranges(script) if script else ())


def read_pack(code, languages_dir=LANGUAGES_DIR):
    """The LanguagePack of the language code in languages_dir.

    A folder without language.json gives a pack that says nothing, so the
    program's defaults apply. A language.json that can't be used raises
    ManifestError, which names the file and the reason.
    """
    folder = Path(languages_dir) / code
    manifest = exact_file(folder, MANIFEST_NAME)
    if manifest is None:
        return LanguagePack(code, folder)
    try:
        data = json.loads(manifest.read_bytes().decode("utf-8"))
        _object(data, "language.json", _MANIFEST_KEYS)
        return LanguagePack(
            code,
            folder,
            _text(data["name"], '"name"') if "name" in data else None,
            _font(data["font"]) if "font" in data else None,
            _punctuation(data["punctuation"]) if "punctuation" in data else None,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ManifestError) as error:
        raise ManifestError(f"{manifest}: {error}") from error


def read_packs(languages_dir=LANGUAGES_DIR):
    """({code: LanguagePack}, [problem, ...]) for every language folder in languages_dir.

    A folder whose language.json can't be used is left out and its problem
    listed, so one broken manifest leaves the other languages as they are.
    """
    packs, problems = {}, []
    for code in language_folders(languages_dir):
        try:
            packs[code] = read_pack(code, languages_dir)
        except ManifestError as error:
            problems.append(str(error))
    return packs, problems
