# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Thread budget for Numba ``parallel=True`` kernels.

Numba sizes its pool from every logical CPU. Long CPU kernels here leave two of
the CPUs this process may run on free instead, as the ``logo`` command does, so
the desktop, the Viewer and other tools stay responsive. A parallel loop also
waits for its slowest thread: on a 20-CPU test machine, using all 20 made these
kernels up to 1.5 times slower than using 18. Setting ``NUMBA_NUM_THREADS``
before start-up overrides the rule; that value is used as given.
"""

from __future__ import annotations

from contextlib import contextmanager
import os

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    NUMBA_AVAILABLE = False


RESERVED_CPUS = 2


def usable_cpu_count() -> int:
    """Return the logical CPUs this process may run on, at least one."""
    counter = getattr(os, "process_cpu_count", None)
    if counter is not None:
        count = counter()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return max(1, int(count or 1))


def choose_thread_count(
    numba_limit: int,
    *,
    logical_cpus: int | None = None,
    explicit_limit: bool = False,
    reserved: int = RESERVED_CPUS,
) -> int:
    """Return the usable CPUs minus ``reserved``, within Numba's pool size."""
    numba_limit = max(1, int(numba_limit))
    if explicit_limit:
        return numba_limit
    if logical_cpus is None:
        logical_cpus = usable_cpu_count()
    budget = max(1, int(logical_cpus) - int(reserved))
    return min(numba_limit, budget)


def default_thread_count() -> int:
    """Return the kernel thread count for this process (1 without Numba)."""
    if not NUMBA_AVAILABLE:
        return 1
    return choose_thread_count(
        numba.config.NUMBA_NUM_THREADS,
        explicit_limit="NUMBA_NUM_THREADS" in os.environ,
    )


@contextmanager
def limited_threads(count: int):
    """Run parallel kernels on ``count`` threads, then restore the setting.

    Numba keeps the setting per calling thread, so this never changes the
    count used by kernels launched from other threads.
    """
    if not NUMBA_AVAILABLE:
        yield
        return
    previous = numba.get_num_threads()
    selected = max(1, min(int(count), numba.config.NUMBA_NUM_THREADS))
    if selected != previous:
        numba.set_num_threads(selected)
    try:
        yield
    finally:
        if selected != previous:
            numba.set_num_threads(previous)


__all__ = [
    "RESERVED_CPUS",
    "NUMBA_AVAILABLE",
    "usable_cpu_count",
    "choose_thread_count",
    "default_thread_count",
    "limited_threads",
]
