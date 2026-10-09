# Bundled desktop Noto fonts

This directory contains the application's compact, offline desktop font core
and a font for Simplified Chinese. Qt registers these files as private
application fonts; the installer does not install them into Windows, macOS,
or Linux.

- `noto/NotoSans/`: Regular, Medium, SemiBold, and Bold hinted static TTFs.
- `noto/NotoSansMono/`: Regular, Medium, SemiBold, and Bold hinted static TTFs.
- `noto/NotoSansSC/`: Regular and Bold Noto Sans SC, cut down to the
  characters of GB2312 (see "Simplified Chinese" below).
- `SHA256SUMS`: the exact declared runtime asset list and checksum of every
  bundled font file.

The eight core fonts come from Noto monthly release
`noto-monthly-release-2026.05.01` (commit
`66c4b351c58f99ace5a6265d329080d74b057909`) and total 4,908,576 bytes
(4.68 MiB). They cover Latin, Greek, Cyrillic, IPA, combining marks, and common
scientific punctuation. Other scripts use fonts installed in the operating
system. Users who need additional coverage should install the appropriate Noto
family and restart the application; copying a font beside these files does not
register it with Qt.

The small WOFF2 assets used by embedded browser pages remain separately under
the parent directory and `docs/fonts/`; KaTeX's mathematical fonts are
unchanged.

## Simplified Chinese

The windows shown in Simplified Chinese use Noto Sans SC from
`noto/NotoSansSC/`. It is registered only while that language shows, so other
languages keep the system's fonts for the same characters, as Japanese does
for kanji. Qt uses it for the characters the core fonts lack. VisPy draws each
text in a single face, so the Viewer draws all its text in it. The browser
views use it too: a page served in Simplified Chinese, and the Tools help
panel, carry @font-face rules for the family
(`Desktop_App.language_web_font_css`), which their font stacks name right
after the core family, as the Qt stacks do.

The two files total 4,583,760 bytes (4.37 MiB). They come from Google Fonts'
variable font `ofl/notosanssc/NotoSansSC[wght].ttf`: version 2.004-H2,
17,772,300 bytes, git blob `fb0637bafbcd804fe32152370a1225990745b4bc`, last
changed in google/fonts commit `2894aab31764f10f29c421bdfd2340d3b382d384`.
fontTools 4.65.0 made each file:

- at weight 400 for Regular, or 700 for Bold;
- with only the 7,445 characters of GB2312, which include all of its 6,763
  hanzi, plus ASCII, Latin-1, Latin Extended-A, General Punctuation, CJK
  Symbols and Punctuation, and Halfwidth and Fullwidth Forms, where the font
  has them;
- with only the layout features Qt applies to horizontal text, and no
  hinting.

`tests/test_application_fonts.py` checks that both files cover GB2312 and
every character of the Simplified Chinese catalog. If a translation needs a
character beyond them, remake the files in a folder that holds the downloaded
variable font. The first command lists GB2312's characters in
`characters.txt`; add the new ones to that file before running the others.

```
python -c "open('characters.txt', 'w', encoding='utf-8').write(''.join(bytes((r, c)).decode('gb2312', 'ignore') for r in range(0xA1, 0xF8) for c in range(0xA1, 0xFF)))"
fonttools varLib.instancer "NotoSansSC[wght].ttf" wght=400 --update-name-table --no-recalc-timestamp -o static.ttf
pyftsubset static.ttf --text-file=characters.txt --unicodes=U+0020-007E,U+00A0-017F,U+2000-206F,U+3000-303F,U+FF00-FFEF --layout-features=ccmp,locl,rlig,liga,clig,calt,rclt,kern,mark,mkmk,curs,rvrn --no-hinting --name-IDs="*" --notdef-outline --recalc-bounds --canonical-order --output-file=NotoSansSC-Regular.ttf
```

Run the last two commands again with `wght=700` and `NotoSansSC-Bold.ttf`,
then update `SHA256SUMS` and the sizes here. Unchanged, these commands give
the bundled files byte for byte.
