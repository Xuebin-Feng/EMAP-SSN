# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Bundle the Lucide icons EMAP-SSN uses, from a downloaded lucide-static package.

    python src/resources/icons/Update_Icons.py <package>   copy the icons and colour the stylesheet's
    python src/resources/icons/Update_Icons.py --colours   only rewrite the stylesheet's coloured copies
    python src/resources/icons/Update_Icons.py --check     change nothing; fail if anything is stale

<package> is the folder an npm tarball of lucide-static unpacks to (it holds
package.json, LICENSE and icons/). README.md says where to get the pinned
version and how to check its integrity.

The update copies every icon in ICONS into lucide/ unchanged, with the
package's LICENSE, and refuses an SVG holding anything but plain shapes (no
scripts, links, styles or text). It then writes the stylesheet's coloured
copies into qss/: Qt draws an SVG's currentColor as black, so each
(icon, token) pair in desktop.Studio_Theme.QSS_ICON_VARIANTS becomes a copy
with that token's colour written in, for every theme colour (THEME_COLORS).
A file that left the list is removed. --colours does only this, from the
bundled icons, as after a change to the theme colours.

--check exits with 1 if an icon in ICONS is missing from lucide/, if lucide/
holds an icon not in ICONS, or if a coloured copy no longer matches its icon
and token. The test suite runs it.

To use a new icon, add its name (as on lucide.dev) to ICONS and rerun the
update with the package of LUCIDE_VERSION.
"""

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

ICONS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ICONS_DIR.parents[1]))  # src, for desktop.Studio_Theme

LUCIDE_VERSION = "1.47.0"

# Every icon the windows and web pages use, by its Lucide name.
ICONS = (
    # Fields, panels and tab bars (the stylesheet's arrows, the hint card, the language row)
    "chevron-down", "chevron-left", "chevron-right", "chevron-up", "folder-open", "globe", "info",
    # Tools cards and help headings
    "dna", "file-input", "file-output", "file-text", "scissors", "search", "settings",
    "sparkles", "timer", "trending-down", "triangle-alert", "wrench",
    # Viewer sidebar
    "bot", "chart-column", "panel-right-close", "panel-right-open",
    # Viewer web pages (agent.html, meta.html, attachments.js)
    "brush-cleaning", "camera", "check", "copy", "download", "eye", "eye-off", "loader-circle", "menu",
    "moon", "mouse-pointer", "plus", "redo-2", "save", "send", "sheet", "sun", "trash-2", "undo-2",
    "upload", "x",
    # Viewer web page esmfold.html (Fold View)
    "crosshair", "layers", "maximize", "palette", "rotate-ccw", "ruler", "tag",
    # The tools' chart windows' toolbar (tools/tool_helpers/Tool_Charts.py; save is above)
    "arrow-left", "arrow-right", "house", "move", "sliders-horizontal", "square-pen", "zoom-in",
    # The Viewer's Console button, at the foot of its window (desktop/Viewer_Widgets.py; the Console
    # window's own buttons use copy, download, trash-2, file-text, folder-open and the chevrons, above)
    "square-terminal",
)

ALLOWED_ELEMENTS = {"svg", "path", "circle", "rect", "line", "polyline", "polygon", "ellipse"}


def _check_plain_svg(path):
    """Refuse an SVG holding anything but plain shapes."""
    root = ElementTree.parse(path).getroot()
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        if tag not in ALLOWED_ELEMENTS:
            raise ValueError(f"{path.name}: unexpected <{tag}> element")
        for attribute in element.attrib:
            name = attribute.split("}")[-1].lower()
            if name.startswith("on") or "href" in name or name == "style":
                raise ValueError(f"{path.name}: unexpected {name} attribute")


def _variants():
    """(path, text) of every coloured copy, once each; theme colours may share one."""
    from desktop.Studio_Theme import (
        QSS_ICON_VARIANTS, THEME_COLORS, coloured_svg, qss_icon_path, theme_tokens,
    )

    made = {}
    for color in THEME_COLORS:
        tokens = theme_tokens(color)
        for name, token in QSS_ICON_VARIANTS:
            path = qss_icon_path(name, token, tokens)
            if path not in made:
                original = (ICONS_DIR / "lucide" / f"{name}.svg").read_text(encoding="utf-8")
                made[path] = coloured_svg(original, tokens[token])
    return made.items()


def write_colours():
    """Write the coloured copies into qss/ and remove any that left the list; returns how many."""
    qss = ICONS_DIR / "qss"
    qss.mkdir(exist_ok=True)
    written = set()
    for path, text in _variants():
        path.write_bytes(text.encode("utf-8"))
        written.add(path.name)
    for stale in qss.glob("*.svg"):
        if stale.name not in written:
            stale.unlink()
    return len(written)


def update(package):
    package = Path(package)
    meta = json.loads((package / "package.json").read_text(encoding="utf-8"))
    if (meta.get("name"), meta.get("version")) != ("lucide-static", LUCIDE_VERSION):
        raise SystemExit(f"Expected lucide-static {LUCIDE_VERSION}, found "
                         f"{meta.get('name')} {meta.get('version')}")
    target = ICONS_DIR / "lucide"
    target.mkdir(exist_ok=True)
    for name in ICONS:
        source = package / "icons" / f"{name}.svg"
        if not source.is_file():
            raise SystemExit(f"The package has no icon named {name!r}")
        _check_plain_svg(source)
        shutil.copyfile(source, target / source.name)
    shutil.copyfile(package / "LICENSE", target / "LICENSE")
    for stale in target.glob("*.svg"):
        if stale.stem not in ICONS:
            stale.unlink()
    written = write_colours()
    print(f"Bundled {len(ICONS)} Lucide {LUCIDE_VERSION} icons and {written} coloured copies.")


def check():
    """Return the list of problems; empty when the bundled icons are up to date."""
    problems = []
    lucide = ICONS_DIR / "lucide"
    bundled = {path.stem for path in lucide.glob("*.svg")}
    problems += [f"missing icon: {name}" for name in ICONS if name not in bundled]
    problems += [f"icon not in ICONS: {name}" for name in sorted(bundled - set(ICONS))]
    if not (lucide / "LICENSE").is_file():
        problems.append("missing lucide/LICENSE")
    for name in sorted(bundled & set(ICONS)):
        try:
            _check_plain_svg(lucide / f"{name}.svg")
        except ValueError as error:
            problems.append(str(error))
    written = set()
    for path, text in _variants():
        written.add(path.name)
        # A checkout may have turned the line endings to CRLF; the bytes otherwise match.
        if not path.is_file() or path.read_bytes().replace(b"\r\n", b"\n") != text.encode("utf-8"):
            problems.append(f"stale coloured copy: qss/{path.name}")
    problems += [f"coloured copy not in QSS_ICON_VARIANTS: qss/{path.name}"
                 for path in (ICONS_DIR / "qss").glob("*.svg") if path.name not in written]
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("package", nargs="?", help="an unpacked lucide-static package")
    parser.add_argument("--colours", action="store_true",
                        help="only rewrite the coloured copies, from the bundled icons")
    parser.add_argument("--check", action="store_true", help="change nothing; fail if stale")
    args = parser.parse_args(argv)
    if args.colours:
        print(f"Wrote {write_colours()} coloured copies.")
        return 0
    if args.check:
        problems = check()
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    if not args.package:
        parser.error("give the unpacked lucide-static package, or --check")
    update(args.package)
    return 0


if __name__ == "__main__":
    sys.exit(main())
