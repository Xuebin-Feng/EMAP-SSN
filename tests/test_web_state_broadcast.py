# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The metadata state sent to the web table (EMAPSSN_Viewer.MainViewer):
get_serializable_metadata builds the same rows as a per-cell loop, and
broadcast_metadata_state builds nothing while no page is connected."""

import json
import os
import sys
import threading
import time
import unittest
import urllib.request
from queue import Queue
from unittest import mock

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from EMAPSSN_Viewer import MainViewer  # noqa: E402
from web_ui import Web_Server  # noqa: E402
from tests.test_web_server import use_private_session_directory  # noqa: E402


def reference_rows(viewer):
    """The rows as the viewer built them one cell at a time, before the
    columns were converted in one pass."""
    rows = []
    for row_idx in range(viewer.n_nodes):
        row_dict = {
            "id": row_idx,
            "Node ID": str(viewer.full_headers[row_idx])
        }
        for key, entry in viewer.metadata.items():
            if key in ("id", "Node ID"):
                continue
            val = entry["values"][row_idx]
            if isinstance(val, (float, np.floating)) and np.isnan(val):
                val = ""
            elif isinstance(val, (float, np.floating)) and np.isinf(val):
                val = "inf" if val > 0 else "-inf"
            else:
                val = val.item() if hasattr(val, 'item') else val
            row_dict[key] = val
        rows.append(row_dict)
    return rows


def reference_event(viewer):
    """The state_updated event as broadcast_metadata_state built it."""
    return {
        "type": "state_updated",
        "visible_mask": viewer.visible_mask.tolist(),
        "selected_indices": viewer.selected_indices,
        "metadata": reference_rows(viewer),
        "columns": ["Node ID"] + list(viewer.metadata.keys()),
        "types": {
            key: entry["type"] for key, entry in viewer.metadata.items()
        },
    }


def column(kind, values):
    return {"type": kind, "values": values}


def objects(*items):
    """An object array holding exactly these items, nested ones included."""
    values = np.empty(len(items), dtype=object)
    for row, item in enumerate(items):
        values[row] = item
    return values


def make_viewer(n_nodes=8):
    """Number and text columns holding NaN, +/-inf, None and numpy scalars in
    object columns, and columns named like the row keys."""
    nan, inf = float("nan"), float("inf")
    viewer = MainViewer.__new__(MainViewer)
    viewer.n_nodes = n_nodes
    viewer.full_headers = [f"node-{i}" for i in range(n_nodes)]
    viewer.visible_mask = np.array([True, False] * (n_nodes // 2))
    viewer.selected_indices = [1, 3]
    viewer.metadata = {
        "f64": column("number", np.array(
            [1.5, nan, inf, -inf, 0.0, -0.0, 3.0, 1e300])),
        "f32": column("number", np.array(
            [0.1, nan, inf, -inf, 2.5, -7.25, 0.0, 1e30], dtype=np.float32)),
        "f16": column("number", np.array(
            [0.5, nan, inf, -inf, 2.0, 1.0, 3.5, 4.0], dtype=np.float16)),
        "all NaN": column("number", np.full(8, nan)),
        "i32": column("number", np.array(
            [1, -2, 3, 4, 5, 6, 7, 2**31 - 1], dtype=np.int32)),
        "i64": column("number", np.array(
            [1, -2, 3, 4, 5, 6, 7, 2**62], dtype=np.int64)),
        "u8": column("number", np.array(
            [0, 1, 2, 3, 4, 5, 6, 255], dtype=np.uint8)),
        "flag": column("text", np.array(
            [True, False, True, True, False, False, True, False])),
        "words": column("text", np.array(
            ["a", "", "bb", "ccc", "dddd", "ü", "ß", "end"])),
        "text": column("text", objects(
            "alpha", None, "", "ü", np.str_("x"), "y", "z", "end")),
        "object numbers": column("number", objects(
            np.int64(5), np.float32(2.5), np.float64(nan), nan, inf, -inf,
            np.float32("inf"), np.float64("-inf"))),
        "object mixed": column("text", objects(
            None, True, 7, 2.5, "s", np.int8(3), np.bool_(True),
            np.array(9))),
        "object nested": column("text", objects(
            [1, 2], {"k": 1}, (3, 4), "s", None, 0, 1.5, np.int16(4))),
        "int ns": column("number", np.array(
            ["2020-01-01"] * 8, dtype="datetime64[ns]")),
        "strided": column("number", np.arange(16.0)[::2]),
        "longer than the nodes": column("number", np.arange(10.0)),
        "plain list": column("text", [
            "a", None, np.float64("nan"), np.int64(3), 4, float("inf"), "g",
            np.float32(1.5)]),
        # Columns that share a row key's name must not overwrite it.
        "id": column("number", np.arange(8) + 100.0),
        "Node ID": column("text", np.array(
            ["x"] * 8, dtype=object)),
        "last": column("text", np.array(list("abcdefgh"), dtype=object)),
    }
    return viewer


class SerializableMetadataTests(unittest.TestCase):
    def assert_same_rows(self, viewer):
        rows = viewer.get_serializable_metadata()
        expected = reference_rows(viewer)

        self.assertEqual(rows, expected)
        # == treats 1 and 1.0 and True alike; the table and JSON do not.
        self.assertEqual(repr(rows), repr(expected))
        self.assertEqual(
            [list(row) for row in rows], [list(row) for row in expected]
        )
        return rows

    def test_rows_equal_the_per_cell_reference_and_their_json(self):
        viewer = make_viewer()

        rows = self.assert_same_rows(viewer)

        self.assertEqual(json.dumps(rows), json.dumps(reference_rows(viewer)))
        # JSON.parse in the browser rejects Infinity and NaN.
        json.dumps(rows, allow_nan=False)

    def test_other_numpy_types_become_what_item_returns(self):
        # Bytes and dates are not JSON, so only the Python values are compared.
        viewer = make_viewer()
        viewer.metadata = {
            "raw": column("text", np.array(
                [b"a", b"", b"bb", b"c", b"d", b"e", b"f", b"g"])),
            "day": column("text", np.array(
                ["2020-01-01"] * 8, dtype="datetime64[D]")),
            "span": column("number", np.arange(8).astype("timedelta64[s]")),
            "complex": column("number", np.arange(8) * (1 + 2j)),
            "records": column("text", np.array(
                [(i, i * 0.5) for i in range(8)],
                dtype=[("a", "i4"), ("b", "f8")])),
            "object bytes": column("text", objects(
                b"a", None, np.bytes_(b"c"), 1, 2, 3, 4, 5)),
        }

        self.assert_same_rows(viewer)

    def test_the_row_keys_come_first_and_cannot_be_overwritten(self):
        viewer = make_viewer()

        rows = self.assert_same_rows(viewer)

        self.assertEqual(list(rows[0])[:3], ["id", "Node ID", "f64"])
        self.assertEqual(list(rows[0])[-1], "last")
        self.assertEqual(sum(key in ("id", "Node ID") for key in rows[0]), 2)
        self.assertEqual([row["id"] for row in rows], list(range(8)))
        self.assertEqual(
            [row["Node ID"] for row in rows],
            [f"node-{i}" for i in range(8)],
        )

    def test_not_a_number_and_infinity_follow_the_table_rules(self):
        viewer = make_viewer()

        rows = viewer.get_serializable_metadata()

        for name in ("f64", "f32", "f16"):
            self.assertEqual(
                [rows[i][name] for i in (1, 2, 3)], ["", "inf", "-inf"], name
            )
        self.assertEqual({row["all NaN"] for row in rows}, {""})
        self.assertEqual(
            [row["object numbers"] for row in rows],
            [5, 2.5, "", "", "inf", "-inf", "inf", "-inf"],
        )
        self.assertEqual(rows[0]["f32"], float(np.float32(0.1)))
        self.assertIs(type(rows[0]["f32"]), float)
        self.assertIs(type(rows[0]["i32"]), int)
        self.assertIs(type(rows[0]["flag"]), bool)
        self.assertIs(type(rows[0]["words"]), str)
        self.assertEqual(rows[1]["text"], None)
        self.assertEqual(rows[7]["longer than the nodes"], 7.0)

    def test_no_nodes_and_no_columns(self):
        viewer = make_viewer()
        viewer.metadata = {}
        self.assert_same_rows(viewer)
        viewer.n_nodes = 0
        viewer.full_headers = []
        viewer.metadata = make_viewer().metadata
        self.assertEqual(viewer.get_serializable_metadata(), [])
        self.assertEqual(reference_rows(viewer), [])

    def test_a_column_shorter_than_the_nodes_still_raises(self):
        for values in (np.arange(5.0), np.array(list("abcde"), dtype=object),
                       list("abcde")):
            with self.subTest(type=type(values).__name__):
                viewer = make_viewer()
                viewer.metadata = {"short": column("text", values)}
                with self.assertRaises(IndexError):
                    reference_rows(viewer)
                with self.assertRaises(IndexError):
                    viewer.get_serializable_metadata()
        viewer = make_viewer()
        viewer.full_headers = viewer.full_headers[:5]
        with self.assertRaises(IndexError):
            viewer.get_serializable_metadata()

    def test_a_large_table_equals_the_reference(self):
        rng = np.random.default_rng(1)
        count = 5000
        viewer = make_viewer(8)
        viewer.n_nodes = count
        viewer.full_headers = [f"WP_{i:09d}" for i in range(count)]
        viewer.visible_mask = np.ones(count, dtype=bool)
        numbers = rng.random(count) * 100
        numbers[rng.random(count) < 0.2] = np.nan
        numbers[rng.random(count) < 0.05] = np.inf
        numbers[rng.random(count) < 0.05] = -np.inf
        viewer.metadata = {
            "score": column("number", numbers),
            "length": column("number", rng.integers(50, 1500, count)),
            "organism": column("text", np.array(
                [f"org {i % 97}" for i in range(count)], dtype=object)),
        }

        rows = self.assert_same_rows(viewer)

        self.assertEqual(
            json.dumps(rows, allow_nan=False),
            json.dumps(reference_rows(viewer), allow_nan=False),
        )


class RecordingLock:
    """A lock that counts how often it was taken."""

    def __init__(self):
        self._lock = threading.Lock()
        self.acquired = 0

    def __enter__(self):
        self.acquired += 1
        return self._lock.__enter__()

    def __exit__(self, *exc_info):
        return self._lock.__exit__(*exc_info)


class FakeWebServer:
    def __init__(self, queues=()):
        self.event_queues = list(queues)
        self.queues_lock = RecordingLock()


class BroadcastMetadataStateTests(unittest.TestCase):
    def silent_viewer(self):
        viewer = make_viewer()
        viewer.broadcast_event = mock.Mock()
        viewer.get_serializable_metadata = mock.Mock(
            side_effect=AssertionError("rows built with no page listening")
        )
        return viewer

    def test_no_server_or_an_empty_queue_list_builds_and_sends_nothing(self):
        viewer = self.silent_viewer()
        self.assertFalse(hasattr(viewer, "web_server"))
        viewer.broadcast_metadata_state()

        viewer.web_server = None
        viewer.broadcast_metadata_state()

        server = FakeWebServer()
        viewer.web_server = server
        viewer.broadcast_metadata_state()

        viewer.get_serializable_metadata.assert_not_called()
        viewer.broadcast_event.assert_not_called()
        # The queues are read under their lock, as broadcast_event reads them.
        self.assertEqual(server.queues_lock.acquired, 1)

    def test_a_connected_page_is_sent_the_event_it_always_was(self):
        viewer = make_viewer()
        first, second = Queue(), Queue()
        viewer.web_server = FakeWebServer([first, second])

        viewer.broadcast_metadata_state()

        for queue in (first, second):
            self.assertEqual(queue.qsize(), 1)
        event = first.get_nowait()
        self.assertIs(second.get_nowait(), event)
        self.assertEqual(event, reference_event(viewer))
        self.assertEqual(
            json.dumps(event), json.dumps(reference_event(viewer))
        )
        self.assertEqual(
            list(event),
            ["type", "visible_mask", "selected_indices", "metadata",
             "columns", "types"],
        )
        self.assertEqual(
            event["columns"], ["Node ID"] + list(viewer.metadata)
        )

    def test_the_event_goes_through_broadcast_event(self):
        viewer = make_viewer()
        viewer.web_server = FakeWebServer([Queue()])
        viewer.broadcast_event = mock.Mock()

        viewer.broadcast_metadata_state()

        viewer.broadcast_event.assert_called_once_with(reference_event(viewer))

    def test_an_emptied_queue_list_stops_the_building_again(self):
        viewer = make_viewer()
        queue = Queue()
        viewer.web_server = FakeWebServer([queue])
        viewer.broadcast_metadata_state()
        self.assertEqual(queue.qsize(), 1)

        viewer.web_server.event_queues.remove(queue)
        viewer.get_serializable_metadata = mock.Mock(
            side_effect=AssertionError("rows built with no page listening")
        )
        viewer.broadcast_metadata_state()

        self.assertEqual(queue.qsize(), 1)


class LateConnectingPageTests(unittest.TestCase):
    """A page that connects after silent broadcasts starts from the current
    state, and a connected page is sent each change."""

    def setUp(self):
        use_private_session_directory(self)
        self.viewer = make_viewer()
        self.server = Web_Server.start_server(self.viewer, preferred_port=0)
        self.addCleanup(Web_Server.stop_server, self.server)
        self.viewer.web_server = self.server
        self.url = (
            f"http://127.0.0.1:{self.server.server_address[1]}/api/events"
        )

    def next_event(self, response):
        while True:
            line = response.readline().decode("utf-8")
            if line.startswith("data: "):
                return json.loads(line[len("data: "):])

    def wait_for_client(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with self.server.queues_lock:
                if self.server.event_queues:
                    return
            time.sleep(0.01)
        self.fail("the page never registered for events")

    def test_the_page_gets_the_state_that_was_never_broadcast(self):
        self.viewer.get_serializable_metadata = mock.Mock(
            wraps=self.viewer.get_serializable_metadata
        )
        self.viewer.metadata["text"]["values"][0] = "changed"
        self.viewer.metadata["f64"]["values"][0] = 42.0
        self.viewer.broadcast_metadata_state()
        self.viewer.get_serializable_metadata.assert_not_called()

        response = urllib.request.urlopen(self.url, timeout=10)
        self.addCleanup(response.close)
        init = self.next_event(response)

        self.assertEqual(init["type"], "init")
        rows = init["data"]["rows"]
        self.assertEqual(rows[0]["text"], "changed")
        self.assertEqual(rows[0]["f64"], 42.0)
        self.assertEqual(rows, json.loads(json.dumps(reference_rows(self.viewer))))

        # With the page connected, each broadcast reaches it again.
        self.wait_for_client()
        self.viewer.metadata["text"]["values"][1] = "again"
        self.viewer.broadcast_metadata_state()
        event = self.next_event(response)
        self.assertEqual(event["type"], "state_updated")
        self.assertEqual(event["metadata"][1]["text"], "again")
        self.assertEqual(
            event, json.loads(json.dumps(reference_event(self.viewer)))
        )


if __name__ == "__main__":
    unittest.main()
