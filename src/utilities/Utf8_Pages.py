# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""Reading a byte range of a UTF-8 log so that no character is cut in two.

A page of a log is a byte range, and a range that ends inside a multi-byte
character would decode its first bytes to U+FFFD and the rest again on the next
page. ``read_utf8_page`` ends a page on a character boundary, so the next
offset is the first byte of the next character, as the Viewer's command output
pages do (Viewer_Command_Portal.read_output). An offset the caller gives that
lies inside a character is read as it is: its stray bytes decode to U+FFFD.
"""

from __future__ import annotations

import codecs


def incomplete_tail(data: bytes) -> int:
    """Number of trailing bytes of ``data`` that begin a character the data does not finish."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    decoder.decode(data, final=False)
    return len(decoder.getstate()[0])


def read_utf8_page(handle, offset: int, limit: int, size: int, growing: bool = False) -> tuple[bytes, bool]:
    """Read up to ``limit`` bytes at ``offset`` of a file of ``size`` bytes, ending on a character.

    Returns ``(data, held_back)``. A character cut off at the end of the page
    is left for the next page, so a log that is still being written gives it
    whole once the rest arrives; the file's end (``size`` is measured before
    the read) counts as such a cut. ``held_back`` is true when ``growing`` (the
    writer may still write) and what was left is a cut character at the end of
    the file: for the caller that is the end of what can be read now.
    When nothing but a cut character is left: with ``growing`` the page is
    empty and the offset does not move; without it the bytes are returned as
    they are, so a file that really ends inside a character still ends. A
    limit under four bytes with more of the file behind it completes the
    character, so a pager always advances.
    """
    handle.seek(offset)
    data = handle.read(limit)
    pending = incomplete_tail(data)
    if not pending:
        return data, False
    at_end = offset + len(data) >= size
    if pending < len(data):
        return data[:-pending], growing and at_end
    if not at_end:
        lead = data[0]
        data += handle.read((4 if lead >= 0xF0 else 3 if lead >= 0xE0 else 2) - pending)
        pending = incomplete_tail(data)
        if not pending or pending < len(data):
            return (data[:-pending] if pending else data), False
    return (b"", True) if growing else (data, False)
