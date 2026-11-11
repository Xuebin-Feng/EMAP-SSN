# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""WorkerTracker.poll (Viewer_Worker_Tracking): worker output reaches the Viewer's command output whole, a character split across two reads included."""

from __future__ import annotations

import codecs
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import Viewer_Worker_Tracking as tracking  # noqa: E402


def tracker_for(path: Path):
    appended = []
    context = mock.Mock()
    context.portal.append_output = lambda request_id, stream, text: appended.append((stream, text))
    streams = {"stdout": 0, "stderr": 0}
    state = mock.Mock(
        path=path, context=context, identity=None, hidden=None, created=time.monotonic(),
        log_offsets=streams,
        log_decoders={name: codecs.getincrementaldecoder("utf-8")("replace") for name in streams},
    )
    return state, appended


class PollOutputTests(unittest.TestCase):
    def poll_in_pieces(self, data: bytes, size: int) -> str:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "job.json"
            log = Path(str(path) + ".stdout")
            state, appended = tracker_for(path)
            for start in range(0, len(data), size):
                with log.open("ab") as handle:
                    handle.write(data[start:start + size])
                tracking.WorkerTracker.poll(state)
            return "".join(text for stream, text in appended if stream == "stdout")

    def test_a_character_split_across_two_reads_comes_out_whole(self):
        for character in ("\u00e9", "\u540d", "\U0001f600"):
            text = "ab" + character + "cd"
            for size in range(1, len(text.encode()) + 1):
                with self.subTest(character=character, size=size):
                    self.assertEqual(self.poll_in_pieces(text.encode(), size), text)

    def test_bytes_that_are_not_utf8_are_still_replaced(self):
        self.assertEqual(self.poll_in_pieces(b"a\xffb\xfe", 1), "a\ufffdb\ufffd")


if __name__ == "__main__":
    unittest.main()
