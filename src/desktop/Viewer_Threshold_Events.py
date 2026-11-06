# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The ``threshold_changed`` event: tells every connected web page what the Viewer's threshold slider is at.

The slider and the threshold box set the score below which edges are not drawn. A page that shows what the
threshold means (a line on a plot, a count) follows it through this one generic event, ``{"type":
"threshold_changed", "value": <score>}``, sent a short while after the last of a burst of changes (a drag sends
dozens a second), carrying the value the slider ended on. It names no page; a page that wants the value when it
opens asks its own route, which reads ``viewer.current_slider_threshold``.
"""

import threading

# A change is announced this long after the last one seen.
COALESCE_SECONDS = 0.1
EVENT_TYPE = "threshold_changed"


class ThresholdAnnouncer:
    """Coalesces the threshold's changes into one event per burst. Costs a socket-free check while no page listens."""

    def __init__(self, viewer, *, delay=COALESCE_SECONDS, timer_factory=threading.Timer):
        self.viewer = viewer
        self.delay = delay
        self.timer_factory = timer_factory
        self.latest = None
        self._timer = None
        self._lock = threading.Lock()

    def _listening(self):
        server = getattr(self.viewer, "web_server", None)
        if not server:
            return False
        with server.queues_lock:
            return bool(server.event_queues)

    def changed(self, value):
        """The threshold is ``value`` now: announce it once, shortly after the last change of a burst."""
        if not self._listening():
            return
        value = float(value)
        with self._lock:
            self.latest = value
            if self._timer is not None:
                return
            self._timer = self.timer_factory(self.delay, self._announce)
            self._timer.daemon = True
            self._timer.start()

    def _announce(self):
        with self._lock:
            self._timer = None
            value = self.latest
        self.viewer.broadcast_event({"type": EVENT_TYPE, "value": value})
