# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Pages of a UTF-8 log (utilities.Utf8_Pages): a page ends on a character boundary, a cursor always advances, and bytes that are not UTF-8 still decode with replacement."""

from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utilities.Utf8_Pages import incomplete_tail, read_utf8_page  # noqa: E402

CHARACTERS = {"two": "\u00e9", "three": "\u540d", "four": "\U0001f600"}


def read_all(data: bytes, limit: int) -> list[tuple[int, str]]:
    """Page through ``data`` as the log readers do; return (next offset, text) per page."""
    handle = io.BytesIO(data)
    offset, pages = 0, []
    while offset < len(data):
        page, _ = read_utf8_page(handle, offset, limit, len(data))
        offset += len(page)
        pages.append((offset, page.decode("utf-8", errors="replace")))
        assert page, "a page must advance the cursor"
    return pages


class IncompleteTailTests(unittest.TestCase):
    def test_counts_the_bytes_of_an_unfinished_character(self):
        encoded = CHARACTERS["four"].encode()
        for cut in range(1, 4):
            self.assertEqual(incomplete_tail(b"ab" + encoded[:cut]), cut)
        self.assertEqual(incomplete_tail(b"ab" + encoded), 0)
        self.assertEqual(incomplete_tail(b"\xff\xfe"), 0)


class PageTests(unittest.TestCase):
    def test_ten_two_byte_characters_in_five_byte_pages_stay_whole(self):
        text = "\u00e9" * 10
        pages = read_all(text.encode(), 5)
        self.assertEqual("".join(page for _, page in pages), text)
        self.assertNotIn("\ufffd", "".join(page for _, page in pages))
        self.assertEqual([offset for offset, _ in pages], [4, 8, 12, 16, 20])

    def test_every_character_length_at_every_split(self):
        for name, character in CHARACTERS.items():
            text = "ab" + character + "cd" + character
            data = text.encode()
            for limit in range(1, len(data) + 2):
                with self.subTest(character=name, limit=limit):
                    pages = read_all(data, limit)
                    self.assertEqual("".join(page for _, page in pages), text)
                    self.assertEqual(pages[-1][0], len(data))

    def test_a_page_ends_where_the_next_character_begins(self):
        data = ("a" + CHARACTERS["three"] + "b").encode()  # a, 3 bytes, b
        handle = io.BytesIO(data)
        page, _ = read_utf8_page(handle, 0, 3, len(data))
        self.assertEqual(page, b"a")
        self.assertEqual(read_utf8_page(handle, 1, 3, len(data))[0], CHARACTERS["three"].encode())

    def test_a_limit_under_four_bytes_completes_the_character(self):
        data = CHARACTERS["four"].encode() + b"x"
        for limit in (1, 2, 3):
            page, _ = read_utf8_page(io.BytesIO(data), 0, limit, len(data))
            self.assertEqual(page, CHARACTERS["four"].encode())

    def test_invalid_bytes_are_replaced_not_held_back(self):
        data = b"ab\xff\xfecd" + b"\x80" * 3 + b"e"
        for limit in range(1, len(data) + 1):
            with self.subTest(limit=limit):
                pages = read_all(data, limit)
                self.assertEqual("".join(page for _, page in pages), data.decode("utf-8", errors="replace"))

    def test_a_finished_file_that_ends_inside_a_character_still_ends(self):
        data = b"ab" + CHARACTERS["three"].encode()[:2]
        pages = read_all(data, 100)
        self.assertEqual(pages, [(2, "ab"), (4, "\ufffd")])

    def test_a_growing_file_leaves_a_cut_character_for_the_next_read(self):
        character = CHARACTERS["four"].encode()
        for cut in range(1, 4):
            with self.subTest(cut=cut):
                written = b"ab" + character[:cut]
                handle = io.BytesIO(written)
                # The content comes first, and the cut character is the end of what can be read now.
                self.assertEqual(read_utf8_page(handle, 0, 100, len(written), growing=True), (b"ab", True))
                # Only the cut character is left: an empty page, the offset where it was.
                self.assertEqual(read_utf8_page(handle, 2, 100, len(written), growing=True), (b"", True))
                # The writer finishes it, and it comes out whole.
                whole = b"ab" + character + b"c"
                handle = io.BytesIO(whole)
                self.assertEqual(read_utf8_page(handle, 2, 100, len(whole), growing=True), (character + b"c", False))

    def test_a_cut_character_before_the_end_of_the_file_is_not_the_end(self):
        data = b"ab" + CHARACTERS["four"].encode() + b"cd"
        self.assertEqual(read_utf8_page(io.BytesIO(data), 0, 4, len(data), growing=True), (b"ab", False))

    def test_a_finished_file_never_holds_a_cut_character_back_as_the_end(self):
        written = b"ab" + CHARACTERS["four"].encode()[:2]
        handle = io.BytesIO(written)
        self.assertEqual(read_utf8_page(handle, 0, 100, len(written)), (b"ab", False))
        self.assertEqual(read_utf8_page(handle, 2, 100, len(written)), (written[2:], False))

    def test_a_growing_file_holds_back_a_cut_character_when_the_limit_is_small(self):
        character = CHARACTERS["three"].encode()
        written = character[:2]
        self.assertEqual(read_utf8_page(io.BytesIO(written), 0, 2, len(written), growing=True), (b"", True))
        self.assertEqual(read_utf8_page(io.BytesIO(written), 0, 2, len(written), growing=False), (written, False))

    def test_an_offset_inside_a_character_reads_its_stray_bytes_as_replacement(self):
        data = ("a" + CHARACTERS["three"] + "b").encode()
        page, _ = read_utf8_page(io.BytesIO(data), 2, 100, len(data))
        self.assertEqual(page.decode("utf-8", errors="replace"), "\ufffd\ufffdb")


if __name__ == "__main__":
    unittest.main()
