# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Output streams (utilities.Output_Streams): pipes and files switch to UTF-8, a terminal keeps its encoding, text a stream cannot encode is escaped instead of raised, and missing, replaced or closed streams are left alone."""

from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest
from unittest import mock


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities.Output_Streams import configure_output_streams  # noqa: E402


class Terminal(io.BytesIO):
    def isatty(self):
        return True


def cp1252_stream(raw=None):
    """A text stream as Python opens a pipe or file on Western Windows."""
    return io.TextIOWrapper(io.BytesIO() if raw is None else raw, encoding="cp1252")


def written(stream, text):
    stream.write(text)
    stream.flush()
    return stream.buffer.getvalue()


class OutputStreamTests(unittest.TestCase):
    def test_streams_escape_what_a_terminal_cannot_show(self):
        pipe = cp1252_stream()
        terminal = cp1252_stream(Terminal())
        configure_output_streams([pipe, terminal, None])
        self.assertEqual(written(pipe, "α"), "α".encode("utf-8"))
        self.assertEqual(written(terminal, "α"), b"\\u03b1")  # a terminal keeps its encoding

    def test_pipes_and_files_write_any_text_as_utf8(self):
        pipe = cp1252_stream()
        configure_output_streams([pipe])
        text = "Résistance α-amylase 中文"
        self.assertEqual(written(pipe, text), text.encode("utf-8"))

    def test_terminal_keeps_text_its_encoding_can_show(self):
        terminal = cp1252_stream(Terminal())
        configure_output_streams([terminal])
        self.assertEqual(written(terminal, "Résistance"), "Résistance".encode("cp1252"))

    def test_text_utf8_cannot_encode_is_escaped(self):
        # A lone surrogate: how os.fsdecode keeps an undecodable POSIX file-name byte.
        pipe = cp1252_stream()
        configure_output_streams([pipe])
        self.assertEqual(written(pipe, "set-\udce9.fasta"), b"set-\\udce9.fasta")

    def test_default_streams_are_the_process_stdout_and_stderr(self):
        stdout, stderr = cp1252_stream(), cp1252_stream()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            configure_output_streams()
        for stream in (stdout, stderr):
            self.assertEqual((stream.encoding, stream.errors), ("utf-8", "backslashreplace"))

    def test_missing_replaced_and_closed_streams_are_left_alone(self):
        class Unreachable(io.TextIOWrapper):
            def isatty(self):
                raise OSError("the console went away")

        replacement = io.StringIO()  # e.g. a test runner's capture: no reconfigure
        closed = cp1252_stream()
        closed.close()
        unreachable = Unreachable(io.BytesIO(), encoding="cp1252")

        configure_output_streams([None, replacement, closed, unreachable])

        self.assertEqual(unreachable.encoding, "cp1252")


if __name__ == "__main__":
    unittest.main()
