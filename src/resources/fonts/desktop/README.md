# Bundled desktop Noto fonts

This directory contains the application's compact, offline desktop font core.
Qt registers these files as private application fonts; the installer does not
install them into Windows, macOS, or Linux.

- `noto/NotoSans/`: Regular, Medium, SemiBold, and Bold hinted static TTFs.
- `noto/NotoSansMono/`: Regular, Medium, SemiBold, and Bold hinted static TTFs.
- `SHA256SUMS`: the exact declared runtime asset list and checksum of every
  bundled font file.

The eight fonts come from Noto monthly release
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
