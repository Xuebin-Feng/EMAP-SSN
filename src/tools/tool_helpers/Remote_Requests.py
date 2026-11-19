# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Several requests in flight for a model that runs on a remote server.

The caller hands over the sequences in order. Worker threads run the requests,
and the calling thread alone receives the results, in input order, so it can
write them to a file one by one.

The number of requests in flight adapts. A plugin that sees the server answer
"too many requests" raises RateLimitedError (only while rate_limits_signalled()
is true, which it is inside the worker threads started here). That halves the
number in flight and sends the same sequence again after a wait, without
counting it as a failed attempt. A run of successes raises the number by one,
never above the limit the caller set.

Any other error ends the run. Results that arrived in order are still written;
requests still in flight are left to finish on their own daemon threads, so the
process can end at once.
"""

import queue
import threading
import time

from tqdm import tqdm


class RateLimitedError(RuntimeError):
    """The server answered "too many requests" (HTTP 429)."""


_thread_state = threading.local()


def rate_limits_signalled():
    """True inside a worker thread of run_in_order.

    A plugin raises RateLimitedError at once when this is true, and keeps its
    own retry loop for the plain one-request-at-a-time path.
    """
    return getattr(_thread_state, "signalled", False)


def notify(text):
    """Print one line without garbling a progress bar, from any thread."""
    tqdm.write(text)


# Successes in a row before the number in flight goes up by one.
SUCCESSES_PER_INCREASE = 10
# Seconds a sequence waits before it is sent again: this times the number of
# times the server has turned it away, but never more than the maximum.
RATE_LIMIT_WAIT_SECONDS = 5.0
RATE_LIMIT_MAX_WAIT_SECONDS = 30.0
# Times one sequence may be turned away before the run ends.
MAX_RATE_LIMIT_RESENDS = 10
# How often the calling thread looks up from waiting (so Ctrl+C is heard).
_POLL_SECONDS = 0.5


def run_in_order(
    items,
    work,
    consume,
    *,
    max_in_flight,
    successes_per_increase=None,
    rate_limit_wait=None,
    max_resends=None,
    report=notify,
):
    """Run work(item) for every item, several at once; consume in input order.

    work(item) runs in a worker thread and returns a result. consume(index,
    item, result) runs in the calling thread, once per item, in input order.
    An exception from work (other than RateLimitedError) or from consume
    ends the run after the results already in order were consumed.
    """
    # None means the module's value, read now so a test can change it.
    if successes_per_increase is None:
        successes_per_increase = SUCCESSES_PER_INCREASE
    if rate_limit_wait is None:
        rate_limit_wait = RATE_LIMIT_WAIT_SECONDS
    if max_resends is None:
        max_resends = MAX_RATE_LIMIT_RESENDS
    items = list(items)
    total = len(items)
    if total == 0:
        return
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be at least 1.")

    tasks = queue.Queue()
    done = queue.Queue()
    stop = threading.Event()

    def worker():
        _thread_state.signalled = True
        while True:
            task = tasks.get()
            if task is None or stop.is_set():
                return
            index, epoch = task
            try:
                result = work(items[index])
            except Exception as error:
                done.put((index, epoch, False, error))
            else:
                done.put((index, epoch, True, result))

    for number in range(min(max_in_flight, total)):
        threading.Thread(
            target=worker, name=f"remote-request-{number}", daemon=True
        ).start()

    limit = max_in_flight
    epoch = 0  # grows each time the limit is lowered
    successes = 0
    next_new = 0  # next never-sent index
    next_write = 0
    running = 0
    waiting = []  # (ready_time, index) of sequences to send again
    resends = {}
    buffered = {}  # finished, not yet written

    def lower_limit():
        nonlocal limit, epoch, successes
        successes = 0
        epoch += 1
        lowered = max(1, limit // 2)
        if lowered != limit:
            limit = lowered
            report(
                "The server asked for fewer requests at once; "
                f"sending {limit} at a time for now."
            )

    def flush_in_order():
        nonlocal next_write
        while next_write in buffered:
            result = buffered.pop(next_write)
            consume(next_write, items[next_write], result)
            next_write += 1

    try:
        while next_write < total:
            now = time.monotonic()
            # Send again what was turned away, then new sequences, while a
            # slot is free and the finished-but-unwritten backlog stays within
            # twice the limit.
            waiting.sort()
            while waiting and waiting[0][0] <= now and running < limit:
                _, index = waiting.pop(0)
                tasks.put((index, epoch))
                running += 1
            while (
                next_new < total
                and running < limit
                and running + len(waiting) + len(buffered) < 2 * limit
            ):
                tasks.put((next_new, epoch))
                next_new += 1
                running += 1

            timeout = _POLL_SECONDS
            if waiting:
                timeout = min(timeout, max(0.0, waiting[0][0] - now))
            try:
                index, sent_epoch, succeeded, payload = done.get(timeout=timeout)
            except queue.Empty:
                continue
            running -= 1

            if succeeded:
                buffered[index] = payload
                successes += 1
                if successes >= successes_per_increase and limit < max_in_flight:
                    successes = 0
                    limit += 1
                    report(f"The server is keeping up; sending {limit} at a time.")
                flush_in_order()
                continue

            if isinstance(payload, RateLimitedError):
                # Several requests sent before one lowering get the same
                # answer; they count as one.
                if sent_epoch == epoch:
                    lower_limit()
                resends[index] = resends.get(index, 0) + 1
                if resends[index] > max_resends:
                    raise RuntimeError(
                        "The server kept answering 'too many requests' "
                        f"({max_resends} times for the same sequence). "
                        "Nothing more was written. Run again later, "
                        "or lower the number of parallel requests."
                    ) from payload
                delay = min(
                    RATE_LIMIT_MAX_WAIT_SECONDS, rate_limit_wait * resends[index]
                )
                waiting.append((time.monotonic() + delay, index))
                continue

            # A real failure: keep what is already in order, then stop.
            flush_in_order()
            raise payload
    finally:
        stop.set()
        for _ in range(min(max_in_flight, total)):
            tasks.put(None)


def run_as_completed(
    items,
    work,
    consume,
    *,
    max_in_flight,
    fatal=(),
    on_start=None,
    successes_per_increase=None,
    rate_limit_wait=None,
    max_resends=None,
    report=notify,
):
    """Run work(item) for every item, several at once; consume each result as it finishes.

    For work whose results are independent of each other (one structure file
    each), so that no finished result is held back for an earlier one.

    work(item) runs in a worker thread and returns a result. consume(index,
    item, result, error) runs in the calling thread, once per finished item, in
    the order they finish: error is None for a result, else the exception work
    raised (a failure of that item alone, which does not end the run).
    on_start(index, item), if given, runs in the calling thread when an item is
    first handed to a worker.

    work raises RateLimitedError when the server answered "too many requests"
    for a limit that clears with time: that halves the number in flight and
    sends the same item again after a wait, without counting it as a failure.
    An error may carry ``max_resends`` to be sent again fewer times than the
    run's limit. A run of successes raises the number by one, never above
    max_in_flight.

    An exception that is an instance of ``fatal`` ends the run, and so does an
    item turned away more often than allowed: nothing more is sent, the items
    already in flight finish and are consumed (their cost is spent already, so
    a result is kept), then the first such error is raised. A max_in_flight of 1
    runs one item at a time, in input order.
    """
    if successes_per_increase is None:
        successes_per_increase = SUCCESSES_PER_INCREASE
    if rate_limit_wait is None:
        rate_limit_wait = RATE_LIMIT_WAIT_SECONDS
    if max_resends is None:
        max_resends = MAX_RATE_LIMIT_RESENDS
    fatal = tuple(fatal)
    items = list(items)
    total = len(items)
    if total == 0:
        return
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be at least 1.")

    tasks = queue.Queue()
    done = queue.Queue()
    stop = threading.Event()

    def worker():
        while True:
            task = tasks.get()
            if task is None or stop.is_set():
                return
            index, epoch = task
            try:
                result = work(items[index])
            except Exception as error:
                done.put((index, epoch, False, error))
            else:
                done.put((index, epoch, True, result))

    for number in range(min(max_in_flight, total)):
        threading.Thread(
            target=worker, name=f"remote-request-{number}", daemon=True
        ).start()

    limit = max_in_flight
    epoch = 0  # grows each time the limit is lowered
    successes = 0
    next_new = 0  # next never-sent index
    finished = 0
    running = 0
    waiting = []  # (ready_time, index) of items to send again
    resends = {}
    stop_error = None

    def send(index, first):
        nonlocal running
        if first and on_start is not None:
            on_start(index, items[index])
        tasks.put((index, epoch))
        running += 1

    def lower_limit():
        nonlocal limit, epoch, successes
        successes = 0
        epoch += 1
        lowered = max(1, limit // 2)
        if lowered != limit:
            limit = lowered
            report(
                "The server asked for fewer requests at once; "
                f"sending {limit} at a time for now."
            )

    try:
        while finished < total:
            if stop_error is not None and running == 0:
                break
            now = time.monotonic()
            if stop_error is None:
                waiting.sort()
                while waiting and waiting[0][0] <= now and running < limit:
                    _, index = waiting.pop(0)
                    send(index, False)
                while next_new < total and running < limit:
                    send(next_new, True)
                    next_new += 1

            timeout = _POLL_SECONDS
            if waiting and stop_error is None:
                timeout = min(timeout, max(0.0, waiting[0][0] - now))
            try:
                index, sent_epoch, succeeded, payload = done.get(timeout=timeout)
            except queue.Empty:
                continue
            running -= 1

            if succeeded:
                successes += 1
                if (
                    stop_error is None
                    and successes >= successes_per_increase
                    and limit < max_in_flight
                ):
                    successes = 0
                    limit += 1
                    report(f"The server is keeping up; sending {limit} at a time.")
                consume(index, items[index], payload, None)
                finished += 1
                continue

            if fatal and isinstance(payload, fatal):
                if stop_error is None:
                    stop_error = payload
                continue

            if isinstance(payload, RateLimitedError):
                if stop_error is not None:
                    continue
                # Several requests sent before one lowering get the same
                # answer; they count as one.
                if sent_epoch == epoch:
                    lower_limit()
                resends[index] = resends.get(index, 0) + 1
                allowed = getattr(payload, "max_resends", max_resends)
                if resends[index] > allowed:
                    error = RuntimeError(
                        "The server kept answering 'too many requests' "
                        f"({allowed} times for the same sequence): {payload}"
                    )
                    error.__cause__ = payload
                    stop_error = error
                    continue
                delay = min(
                    RATE_LIMIT_MAX_WAIT_SECONDS, rate_limit_wait * resends[index]
                )
                waiting.append((time.monotonic() + delay, index))
                continue

            consume(index, items[index], None, payload)
            finished += 1
        if stop_error is not None:
            raise stop_error
    finally:
        stop.set()
        for _ in range(min(max_in_flight, total)):
            tasks.put(None)
