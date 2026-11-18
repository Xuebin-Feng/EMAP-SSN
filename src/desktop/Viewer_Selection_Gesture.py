# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""A selection gesture that something outside the Viewer's canvas drives, as a box selection is driven from the canvas.

A web page that draws its own selection box (Tree View does) can make the network's selection follow the box while
the button is still down, as the network's own box makes the pages follow it. The page names the gesture with an id
and numbers its updates; the Viewer keeps what is needed to put things right:

* ``update``: the nodes in the box now, and whether they replace the selection the gesture started with (``toggle``
  False) or are toggled against it (``toggle`` True, exact against that start, never against the last update). The
  start is the selection when the first update was taken, which is also what ``cancel`` puts back. An update that is
  not newer than the last one taken, or that belongs to a gesture that has ended, changes nothing.
* ``alive``: the page is still there. A gesture that hears nothing for ``SILENCE_MS`` is cancelled, so a page that
  was closed or lost its connection in the middle of a drag does not leave the Viewer in an open gesture.
* ``end``: the page let go. The message carries the final nodes, so that the result never depends on every update
  having arrived: it is what one plain request of those nodes would have made (the same selection, the status line,
  one step for the commands: selection_epoch is one more than at the start, whatever the updates did), and the
  pages are told if they have not heard it. A request that named nodes none of which can be selected changes nothing.
* ``cancel``: the selection from before the gesture comes back, as Esc does for the network's own box.

While a gesture is open it counts as a mouse gesture (Command_Queue.gesture_open): the commands and the web pages'
other actions wait for it, as they wait for a box drawn on the canvas, and Esc or Ctrl+Z on the canvas, or a press on
it, cancels it. Under Low Resource Mode's Update Colors After Mouse Release the updates change nothing and the end
makes the selection, and under Update Web Pages After Mouse Release the pages hear of it once, at the end, exactly
as for the network's box.

The selection_changed events this makes carry ``"gesture": {"id": ..., "final": true}`` (``final`` only on the one
that ends or cancels the gesture) before the indices, so that the page that drives the gesture can tell its own
echoes from other news. Nothing here knows which page that is.
"""

import collections

import numpy as np

# How long an open gesture may hear nothing before it is cancelled: the page sends ``alive`` about every second.
SILENCE_MS = 5000
# How many ended gestures are remembered, so that their late messages change nothing.
REMEMBERED = 64

PHASES = ("update", "alive", "end", "cancel")


def selection_gesture_of(viewer):
    """The Viewer's selection gesture, made on first use."""
    gesture = getattr(viewer, "outside_selection_gesture", None)
    if gesture is None:
        gesture = SelectionGesture(viewer)
        viewer.outside_selection_gesture = gesture
    return gesture


def gesture_is_open(viewer):
    """Whether a gesture driven from outside is open on ``viewer``."""
    gesture = getattr(viewer, "outside_selection_gesture", None)
    return gesture is not None and gesture.is_open


def _unique(nodes):
    return np.unique(np.asarray(nodes, dtype=np.int64))


class SelectionGesture:
    """The state of the one gesture that may be open (see the module docstring)."""

    def __init__(self, viewer, silence_ms=SILENCE_MS):
        self.viewer = viewer
        self.silence_ms = silence_ms
        self.id = None
        self.closed = collections.deque(maxlen=REMEMBERED)
        self._timer = None
        self._reset()

    def _reset(self):
        self.id = None
        self.base = None            # the selection the gesture started with (sorted, unique)
        self.epoch = 0              # selection_epoch at the start
        self.last_seq = 0
        self.heard = None           # what the pages last heard (sorted, unique)

    @property
    def is_open(self):
        return self.id is not None

    def owns(self, gesture_id):
        return self.id is not None and self.id == gesture_id

    # ---- what the page's messages do ---------------------------------------------------------------------

    def drive(self, gesture_id, phase, seq=0, nodes=None, toggle=False):
        """Take one message of gesture ``gesture_id``; returns {"ok", "changed", "selected", "note"} like a plain request.

        ``nodes`` is the sorted array of the node numbers in the box (in range), for ``update`` and ``end``.
        """
        if gesture_id in self.closed:
            return self._answer(False, "gesture_closed")
        if phase == "cancel":
            if self.owns(gesture_id):
                self.cancel()
            else:
                self.closed.append(gesture_id)
            return self._answer(False, "cancelled")
        if phase == "alive":
            if not self.owns(gesture_id):
                return self._answer(False, "no_gesture")
            self._touch()
            return self._answer(False, "alive")
        if self.is_open and not self.owns(gesture_id):
            self.cancel()                   # another gesture begins: the one that was open is over
        if phase == "update":
            return self._update(gesture_id, seq, nodes, toggle)
        return self._end(gesture_id, nodes, toggle)

    def _answer(self, changed, note):
        selected = len(getattr(self.viewer, "selected_indices", None) or [])
        return {"ok": True, "changed": changed, "selected": selected, "note": note}

    def _open(self, gesture_id):
        viewer = self.viewer
        self.id = gesture_id
        self.base = _unique(getattr(viewer, "selected_indices", None) or [])
        self.epoch = getattr(viewer, "selection_epoch", 0)
        self.last_seq = 0
        self.heard = self.base

    def _result(self, requested, toggle):
        """The selection the nodes in the box make: the hidden ones are left out, as the network's box leaves them out."""
        viewer = self.viewer
        visible = np.asarray(viewer.visible_mask, dtype=bool)
        wanted = requested[visible[requested]] if requested.size else requested
        return wanted, (np.setxor1d(self.base, wanted) if toggle else wanted)

    def _update(self, gesture_id, seq, nodes, toggle):
        if not self.is_open:
            self._open(gesture_id)
        elif seq <= self.last_seq:
            return self._answer(False, "stale")
        self.last_seq = seq
        self._touch()
        requested = _unique(nodes)
        from desktop.Viewer_State import low_resource_drawing, low_resource_webserver
        import EMAPSSN_Config as cfg
        if low_resource_drawing(cfg):
            # Nothing is drawn until the button is let go; the end carries the nodes.
            return self._answer(False, "held")
        _wanted, result = self._result(requested, toggle)
        changed = self._apply(result, broadcast=not low_resource_webserver(cfg))
        return self._answer(changed, None)

    def _end(self, gesture_id, nodes, toggle):
        if not self.is_open:
            self._open(gesture_id)
        viewer = self.viewer
        requested = _unique(nodes)
        wanted, result = self._result(requested, toggle)
        if requested.size and not wanted.size:
            result = self.base                      # nothing that was named can be selected: nothing changes
        final_id = self.id
        base, epoch, heard = self.base, self.epoch, self.heard
        self._apply(result, broadcast=False)
        net_change = not np.array_equal(result, base)
        viewer.selection_epoch = epoch + 1 if net_change else epoch
        if not np.array_equal(result, heard):
            self._tell(final_id, result, final=True)
        self._close()
        if net_change:
            show = getattr(viewer, "_show_selection_status", None)
            if callable(show):
                show()
        self._resume()
        return self._answer(net_change, None if net_change else "unchanged")

    def cancel(self):
        """Put the selection from before the gesture back, and end it (the page's cancel, Esc, a press, a silent page)."""
        if not self.is_open:
            return False
        viewer = self.viewer
        base, epoch, heard, gesture_id = self.base, self.epoch, self.heard, self.id
        self._apply(base, broadcast=False)
        viewer.selection_epoch = epoch                 # as if the gesture had not been made
        if not np.array_equal(base, heard):
            self._tell(gesture_id, base, final=True)
        self._close()
        self._resume()
        return True

    # ---- the Viewer's side of it -----------------------------------------------------------------------------

    def _current(self):
        return _unique(getattr(self.viewer, "selected_indices", None) or [])

    def _apply(self, result, broadcast):
        """Make ``result`` the selection if it is not already: noted as made by hand, drawn, and told to the pages if asked."""
        viewer = self.viewer
        if np.array_equal(result, self._current()):
            return False
        viewer.selected_indices = result.tolist()
        viewer._note_hand_selection()
        viewer.update_selection_visual(broadcast=False)
        if broadcast:
            self._tell(self.id, result, final=False)
        return True

    def _tell(self, gesture_id, result, final):
        info = {"id": gesture_id}
        if final:
            info["final"] = True
        self.viewer.broadcast_event({
            "type": "selection_changed", "gesture": info, "indices": self.viewer.selected_indices})
        self.heard = result

    def _close(self):
        if self._timer is not None:
            self._timer.stop()
        self.closed.append(self.id)
        self._reset()

    def _resume(self):
        resume = getattr(self.viewer, "_resume_command_queue", None)
        if callable(resume):
            resume()

    # ---- a page that went away -------------------------------------------------------------------------------

    def _touch(self):
        if self._timer is None:
            from PySide6 import QtCore
            if QtCore.QCoreApplication.instance() is None:
                return
            self._timer = QtCore.QTimer()
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self._silent)
        self._timer.start(self.silence_ms)

    def _silent(self):
        self.cancel()
