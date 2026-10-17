# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Writes docs/assets/emapssn_logo.png, the EMAP-SSN mark of the documentation pages.

The pages show it at 28 px in the top bar and use it as their favicon, and they
must stay self-contained, so they carry this small copy of the application's
logo, src/bin/logos/viewer_logo.png (4320 x 4320 px, 460 KB), instead of
linking it: scaled down to 64 x 64 px, which is sharp at 2x, with Pillow's
LANCZOS filter and an optimised PNG. To rebuild it, for example after the logo
changes:

    python docs/assets/Make_Logo.py

tests/test_docs_style.py scales the logo again and compares it with the file.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "bin" / "logos" / "viewer_logo.png"
TARGET = Path(__file__).resolve().parent / "emapssn_logo.png"
SIZE = 64


def downscaled(source=SOURCE):
    """The logo as the pages carry it: SIZE x SIZE, RGBA, no metadata."""
    with Image.open(source) as image:
        return image.convert("RGBA").resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def main():
    downscaled().save(TARGET, format="PNG", optimize=True)
    print(f"{TARGET.relative_to(ROOT)}: {TARGET.stat().st_size} bytes, from {SOURCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
