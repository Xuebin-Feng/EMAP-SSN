# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Vendored web libraries keep their upstream bytes, and Mol* stays offline.

THIRD_PARTY_LICENSES.md says these files are copied unchanged, apart from the
attribution banner on Tabulator's two files. One value in Tabulator's stylesheet
once changed by accident (16px became 166px) and nothing noticed. A Windows
checkout may give the files CRLF line ends, so they are compared as git stores
them, with LF. The Mol* bundle has its own recorded hashes and a -text rule.
"""
import hashlib
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"

# Path: (lines of project banner before the upstream body, the upstream file's
# SHA-256 from jsDelivr: tabulator-tables@6.2.1, marked@18.0.9, katex@0.16.8).
VENDORED = {
    "resources/meta/tabulator.min.js": (3, "1d9b9547e9116a5faf74ba96af969884cd674ee2a6055905a119e0c19029f1ae"),
    "resources/meta/tabulator.min.css": (3, "a46d8051944c745cae8a7976b4fb9d93d894d20876a4521cc4f6f035cfef52ea"),
    "resources/agent/marked.umd.js": (0, "ba65f1c8948e6b01321399800843e9048b31e1c197652d4b0fafae840b30e32b"),
    "resources/katex.min.js": (0, "d4f0ea25c4ccb7986b229d6427794b7063cf3209b0376cf5a64bfc4dd7918c95"),
    "resources/katex.min.css": (0, "df9ef84e97d38557de02183e238f8ddfd10988b37741492e12c3157bc856011e"),
    "resources/katex-auto-render.min.js": (0, "9cb8dacfc086c2966c9ec4ba54f4a2dc43b7cbe2b33cec1a2743d886c7fb47a7"),
}


def upstream_body(path: Path, banner_lines: int) -> bytes:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return b"".join(data.splitlines(keepends=True)[banner_lines:])


class VendoredLibraryTests(unittest.TestCase):
    def test_vendored_libraries_keep_their_upstream_bytes(self):
        for relative, (banner_lines, digest) in VENDORED.items():
            with self.subTest(file=relative):
                body = upstream_body(SRC / relative, banner_lines)
                self.assertEqual(hashlib.sha256(body).hexdigest(), digest)

    def test_fold_view_does_not_fetch_the_mol_star_remote_state_list(self):
        # Without this option Mol* asks webchem.ncbr.muni.cz for its list of
        # shared states on every load; THIRD_PARTY_LICENSES.md §2 says the page
        # makes no such request.
        # The option's own line, not the comment that explains it.
        page = (SRC / "web_ui" / "esmfold.html").read_text(encoding="utf-8")
        self.assertRegex(page, r"(?m)^\s*layoutShowRemoteState\s*:\s*false\s*,?\s*$")
        self.assertNotRegex(page, r"(?m)^\s*layoutShowRemoteState\s*:\s*true")


if __name__ == "__main__":
    unittest.main()
