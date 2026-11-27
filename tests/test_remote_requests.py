"""Tests for Remote_Requests.run_as_completed: several requests in flight, each result consumed as it finishes.

run_in_order, the same module's other function, is tested in test_embedding_hdf5_metadata.py. Nothing here
sends a request: ``work`` is a function of the test.
"""

import sys
import threading
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tools.tool_helpers import Remote_Requests
from tools.tool_helpers.Remote_Requests import RateLimitedError, run_as_completed


class Stop(Exception):
    """A fatal error of these tests."""


class Collector:
    """What consume saw, and on which thread."""

    def __init__(self):
        self.results = []
        self.threads = set()

    def __call__(self, index, item, result, error):
        self.threads.add(threading.current_thread())
        self.results.append((index, item, result, error))

    def items(self):
        return [entry[1] for entry in self.results]


class RunAsCompletedTests(unittest.TestCase):
    def run_it(self, items, work, consume=None, **options):
        consume = consume or Collector()
        options.setdefault("max_in_flight", 1)
        options.setdefault("rate_limit_wait", 0.001)
        options.setdefault("report", lambda text: None)
        run_as_completed(items, work, consume, **options)
        return consume

    def test_one_request_at_a_time_runs_in_input_order_and_consumes_on_the_calling_thread(self):
        worked = []
        collector = self.run_it(["a", "b", "c"], lambda item: worked.append(item) or item.upper(), max_in_flight=1)

        self.assertEqual(worked, ["a", "b", "c"])
        self.assertEqual(collector.results, [(0, "a", "A", None), (1, "b", "B", None), (2, "c", "C", None)])
        self.assertEqual(collector.threads, {threading.current_thread()})

    def test_several_requests_are_in_flight_at_once_and_never_more_than_the_limit(self):
        barrier = threading.Barrier(3, timeout=10)
        # Three requests wait for each other: they can only pass the barrier if all three are in flight.
        collector = self.run_it(["a", "b", "c"], lambda item: (barrier.wait(), item)[1], max_in_flight=3)
        self.assertEqual(sorted(collector.items()), ["a", "b", "c"])
        self.assertEqual([entry[3] for entry in collector.results], [None, None, None])

        lock = threading.Lock()
        state = {"now": 0, "most": 0, "finished_others": 0}
        all_others_done = threading.Event()

        def work(item):
            with lock:
                state["now"] += 1
                state["most"] = max(state["most"], state["now"])
            try:
                if item == 0:
                    # Stay in flight while items 1..7 run one by one on the second slot.
                    self.assertTrue(all_others_done.wait(timeout=5.0))
            finally:
                with lock:
                    state["now"] -= 1
                    if item != 0:
                        state["finished_others"] += 1
                        if state["finished_others"] == 7:
                            all_others_done.set()
            return item

        collector = self.run_it(list(range(8)), work, max_in_flight=2)
        self.assertEqual(sorted(collector.items()), list(range(8)))
        self.assertEqual(state["most"], 2)

    def test_a_failed_item_is_consumed_as_an_error_and_does_not_end_the_run(self):
        def work(item):
            if item == "b":
                raise ValueError("this one failed")
            return item

        collector = self.run_it(["a", "b", "c"], work, max_in_flight=2)

        by_item = {entry[1]: entry for entry in collector.results}
        self.assertEqual(by_item["a"][2:], ("a", None))
        self.assertEqual(by_item["c"][2:], ("c", None))
        self.assertIsNone(by_item["b"][2])
        self.assertEqual(str(by_item["b"][3]), "this one failed")

    def test_a_per_minute_answer_lowers_the_number_and_sends_the_same_item_again(self):
        calls = {"b": 0}
        reports = []

        def work(item):
            if item == "b":
                calls["b"] += 1
                if calls["b"] == 1:
                    raise RateLimitedError("slow down")
            return item

        collector = self.run_it(["a", "b", "c", "d"], work, max_in_flight=4, report=reports.append)

        self.assertEqual(calls["b"], 2)
        self.assertEqual(sorted(collector.items()), ["a", "b", "c", "d"])
        self.assertTrue(all(entry[3] is None for entry in collector.results), collector.results)
        self.assertIn("The server asked for fewer requests at once; sending 2 at a time for now.", reports)

    def test_the_number_goes_up_again_after_a_run_of_successes(self):
        reports = []
        state = {"raised": False}

        def work(item):
            if item == 0 and not state["raised"]:
                state["raised"] = True
                raise RateLimitedError("slow down")
            return item

        self.run_it(list(range(12)), work, max_in_flight=4, successes_per_increase=2, report=reports.append)

        self.assertTrue(any("sending 3 at a time" in text for text in reports), reports)

    def test_an_item_the_server_keeps_turning_away_ends_the_run_with_its_text(self):
        calls = []

        def work(item):
            calls.append(item)
            error = RateLimitedError("unclear limit text")
            error.max_resends = 2
            raise error

        with self.assertRaises(RuntimeError) as caught:
            self.run_it(["a"], work, max_in_flight=1)

        self.assertEqual(len(calls), 3, "sent once, then again twice")
        self.assertIn("unclear limit text", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RateLimitedError)

    def test_a_fatal_error_stops_sending_keeps_what_was_in_flight_and_is_raised_last(self):
        sent = []
        b_started = threading.Event()
        a_fatal_seen = threading.Event()

        class _FatalStopMatcherMeta(type):
            def __instancecheck__(cls, instance):
                # Use a distinct matcher type so CPython invokes __instancecheck__
                # instead of short-circuiting on exact type identity.
                if isinstance(instance, Stop):
                    a_fatal_seen.set()
                    return True
                return False

        class FatalStopMatcher(metaclass=_FatalStopMatcherMeta):
            pass

        def work(item):
            sent.append(item)
            if item == "a":
                self.assertTrue(b_started.wait(timeout=5.0))
                raise Stop("limit reached")
            if item == "b":
                b_started.set()
                # Release b only after run_as_completed has received a's fatal error.
                self.assertTrue(a_fatal_seen.wait(timeout=5.0))
            return item

        collector = Collector()
        with self.assertRaises(Stop) as caught:
            self.run_it(
                ["a", "b", "c", "d"],
                work,
                collector,
                max_in_flight=2,
                fatal=(FatalStopMatcher,),
            )

        self.assertEqual(str(caught.exception), "limit reached")
        # b was in flight when a failed: its result is consumed, c and d are never sent.
        self.assertEqual(collector.results, [(1, "b", "b", None)])
        self.assertEqual(sorted(sent), ["a", "b"])

    def test_on_start_runs_on_the_calling_thread_once_for_each_item_even_when_it_is_sent_again(self):
        started = []
        threads = set()
        state = {"raised": False}

        def work(item):
            if item == "b" and not state["raised"]:
                state["raised"] = True
                raise RateLimitedError("slow down")
            return item

        def on_start(index, item):
            started.append((index, item))
            threads.add(threading.current_thread())

        self.run_it(["a", "b", "c"], work, max_in_flight=1, on_start=on_start)

        self.assertEqual(started, [(0, "a"), (1, "b"), (2, "c")])
        self.assertEqual(threads, {threading.current_thread()})

    def test_nothing_to_do_and_a_limit_below_one(self):
        self.assertEqual(self.run_it([], lambda item: item).results, [])
        with self.assertRaises(ValueError):
            run_as_completed(["a"], lambda item: item, Collector(), max_in_flight=0)

    def test_an_error_of_consume_ends_the_run(self):
        def consume(index, item, result, error):
            raise OSError("disk full")

        with self.assertRaises(OSError):
            self.run_it(["a", "b"], lambda item: item, consume, max_in_flight=2)

    def test_run_in_order_is_still_there(self):
        self.assertTrue(callable(Remote_Requests.run_in_order))


if __name__ == "__main__":
    unittest.main()
