# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""A VisPy timer that fires when a test says so, for the Viewer's one-shot timers.

The Viewer waits out a double click before it copies a header, and a second after the last
Ctrl + wheel notch before it ends a turn. A test patches vispy.app.Timer with FakeTimer, lets
the Viewer start its timers, and fires them (or does not) instead of waiting.
"""


class FakeTimer:
    """Stands in for vispy.app.Timer: it keeps its interval and calls its callback on fire().

    Like VisPy's timer, start() does nothing while the timer runs, and a timer with one iteration
    stops once it has fired.
    """

    created = []

    def __init__(self, interval="auto", connect=None, iterations=-1, start=False, app=None):
        self.interval = interval
        self.callback = connect
        self.running = False
        self.starts = []  # the interval of each start that took
        FakeTimer.created.append(self)

    def start(self, interval=None, iterations=None):
        if self.running:
            return
        if interval is not None:
            self.interval = interval
        self.running = True
        self.starts.append(self.interval)

    def stop(self):
        self.running = False

    def fire(self):
        """The time has come: stop, then call the callback with no event, as the timer does with its own."""
        assert self.running, "the timer is not running, so nothing would fire"
        self.running = False
        self.callback(None)
