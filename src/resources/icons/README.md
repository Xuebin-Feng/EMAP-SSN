# Bundled icons

The desktop windows (Config, VR Config, Tools, Viewer) and the Viewer's web
pages draw their icons from [Lucide](https://lucide.dev), the line-icon set of
shadcn/ui: 24 px grid, 2 px round strokes, coloured through `currentColor`.

| Folder | Contents |
|---|---|
| `lucide/` | The icons the program uses, copied unchanged from the package, with Lucide's `LICENSE` (ISC; icons derived from Feather are also MIT). |
| `qss/` | Copies of the stylesheet's arrows with a theme colour written in, named `<icon>-<colour>.svg`. Qt draws `currentColor` as black, so the stylesheet can't colour an icon itself. |
| `Update_Icons.py` | Copies the icons listed in its `ICONS` from a lucide-static package and writes the coloured copies; `--check` reports anything stale. |

## Source

- Package: `lucide-static` 1.47.0 (released 2026-09-17), ISC licence.
- Tarball: `https://registry.npmjs.org/lucide-static/-/lucide-static-1.47.0.tgz`
  (6,786,649 bytes). Its sha512 must match the `dist.integrity` npm lists for
  the version (`https://registry.npmjs.org/lucide-static`), which it did when
  the icons were bundled on 2026-10-10.
- Only plain shapes are accepted: `Update_Icons.py` refuses an SVG holding
  scripts, links, styles or text.

## Adding an icon

1. Find its name on lucide.dev, and add it to `ICONS` in `Update_Icons.py`.
2. Download the tarball above, check its sha512, and unpack it into a folder of its own.
3. Run `python src/resources/icons/Update_Icons.py <folder>/package`, then
   `python src/resources/icons/Update_Icons.py --check`.

In Qt code, `desktop.Studio_Theme.studio_icon(name)` returns the icon in a theme
colour, and `IconLabel(name)` shows it in a label. Web pages inline the SVG markup,
so that CSS `color` sets the stroke.
