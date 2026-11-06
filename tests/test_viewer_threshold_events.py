"""desktop.Viewer_Threshold_Events: the generic threshold_changed event, coalesced per burst of changes."""

import pathlib
import queue
import sys
import threading
import time
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from desktop.Viewer_Threshold_Events import COALESCE_SECONDS, EVENT_TYPE, ThresholdAnnouncer


class FakeTimer:
    """A timer that fires when the test says so."""

    instances = []

    def __init__(self, delay, function):
        self.delay, self.function, self.daemon, self.started = delay, function, False, False
        FakeTimer.instances.append(self)

    def start(self):
        self.started = True


def listening_viewer(clients=1):
    sent = []
    viewer = SimpleNamespace(
        web_server=SimpleNamespace(queues_lock=threading.Lock(), event_queues=[object()] * clients),
        broadcast_event=sent.append,
    )
    return viewer, sent


class ThresholdAnnouncerTests(unittest.TestCase):
    def setUp(self):
        FakeTimer.instances.clear()

    def announcer(self, viewer, **options):
        return ThresholdAnnouncer(viewer, timer_factory=FakeTimer, **options)

    def test_a_burst_of_changes_is_one_event_with_the_last_value(self):
        viewer, sent = listening_viewer()
        announcer = self.announcer(viewer)
        for value in (1.0, 2.5, 4.0):
            announcer.changed(value)
        self.assertEqual(len(FakeTimer.instances), 1)
        timer = FakeTimer.instances[0]
        self.assertTrue(timer.started and timer.daemon)
        self.assertEqual(timer.delay, COALESCE_SECONDS)
        self.assertEqual(sent, [])
        timer.function()
        self.assertEqual(sent, [{"type": EVENT_TYPE, "value": 4.0}])
        # The next change starts another burst, even to a value announced before.
        announcer.changed(1.0)
        self.assertEqual(len(FakeTimer.instances), 2)
        FakeTimer.instances[1].function()
        self.assertEqual(sent[-1], {"type": EVENT_TYPE, "value": 1.0})

    def test_the_event_carries_a_plain_number(self):
        import numpy as np

        viewer, sent = listening_viewer()
        announcer = self.announcer(viewer)
        announcer.changed(np.float32(0.25))
        FakeTimer.instances[0].function()
        self.assertIs(type(sent[0]["value"]), float)

    def test_nothing_is_done_while_no_page_listens(self):
        for viewer in (SimpleNamespace(web_server=None), SimpleNamespace(), listening_viewer(clients=0)[0]):
            announcer = self.announcer(viewer)
            announcer.changed(3.0)
            self.assertEqual(FakeTimer.instances, [])

    def test_the_real_timer_sends_after_the_delay(self):
        viewer, sent = listening_viewer()
        announcer = ThresholdAnnouncer(viewer, delay=0.02)
        announcer.changed(0.5)
        announcer.changed(0.75)
        deadline = time.monotonic() + 2
        while not sent and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(sent, [{"type": EVENT_TYPE, "value": 0.75}])


if __name__ == "__main__":
    unittest.main()
