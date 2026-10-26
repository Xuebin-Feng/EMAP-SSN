# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The requests an agent can make of a process, and their answers.

A request is a JSON file in <out>/requests/ named ``pid-<n>--<id>.json`` or ``kind-<kind>--<id>.json``; the
process it names finds it within about 100 ms, claims it by renaming it, and writes <out>/responses/<id>.json.
Every request has an "action"; most take a "window" (a title regex, or {"title", "class", "objectName"}, or an
index from `windows`) and a "widget" (see targets.py). The actions are the ``do_`` methods of Actions below.

Actions that act on the program like a user (click, type, key, set_value, close, hover, resize, viewer_command)
run from a zero-delay timer, not inside the polling call. The program may answer one with a modal dialog, which
runs a nested event loop and would not return; the polling timer keeps firing inside that loop, so the agent can
capture the dialog and click its buttons, and the answer to the opening request is written after ``block_ms``
(with "completed": false) when the action has not returned by then.

Only the GUI thread calls into this module.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from emapssn_desktop_probe import store, targets
from emapssn_desktop_probe.targets import ActionError

Qt = QtCore.Qt
SETTLE_MS = 150           # how long the program is left to react before the answer to a click or a key is written
VIEWER_SETTLE_MS = 400    # the same for a Viewer command, which may print and update a good deal
BLOCK_MS = 1500           # how long an action may run before it is answered as "not completed"
MODIFIERS = {"ctrl": Qt.KeyboardModifier.ControlModifier, "control": Qt.KeyboardModifier.ControlModifier,
             "shift": Qt.KeyboardModifier.ShiftModifier, "alt": Qt.KeyboardModifier.AltModifier,
             "meta": Qt.KeyboardModifier.MetaModifier, "cmd": Qt.KeyboardModifier.MetaModifier}
BUTTONS = {"left": Qt.MouseButton.LeftButton, "right": Qt.MouseButton.RightButton, "middle": Qt.MouseButton.MiddleButton}
PRINTED_LINES = 40
WRITING_SECONDS = 1.0     # a request file that does not parse yet and is younger than this may still be being written


class Deferred:
    """An action that runs from a timer: ``run()`` does it, ``after(result)`` completes the answer once the program has settled."""

    def __init__(self, run, *, settle_ms=SETTLE_MS, after=None, block_ms=BLOCK_MS):
        self.run = run
        self.settle_ms = settle_ms
        self.after = after
        self.block_ms = block_ms


class Pending:
    def __init__(self, request_id, request, deferred, window_note=None):
        self.request_id = request_id
        self.request = request
        self.deferred = deferred
        self.window_note = window_note
        self.deadline = time.monotonic() + deferred.block_ms / 1000
        self.finished_at = None
        self.result = None
        self.error = None
        self.responded = False


def json_safe(value, depth=0):
    """``value`` as something JSON can hold: basic types as they are, containers walked, anything else as its repr."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth > 6:
        return repr(value)[:200]
    if isinstance(value, dict):
        return {str(key): json_safe(item, depth + 1) for key, item in list(value.items())[:200]}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item, depth + 1) for item in list(value)[:200]]
    if isinstance(value, QtCore.QRect):
        return [value.x(), value.y(), value.width(), value.height()]
    if isinstance(value, (QtCore.QPoint, QtCore.QSize)):
        return [value.x() if hasattr(value, "x") else value.width(), value.y() if hasattr(value, "y") else value.height()]
    return repr(value)[:500]


class Actions:
    def __init__(self, probe):
        self.probe = probe
        self.recorder = probe.recorder
        self._pending: list[Pending] = []
        self._window_note = None  # set when a request's "window" matched more than one window
        self._requests = Path(self.recorder.out) / "requests"
        self._claimed = self.recorder.folder / "claimed"
        self._prefixes = (f"pid-{self.recorder.pid}--", f"kind-{self.recorder.kind}--")

    # --- the inbox
    def poll(self) -> None:
        try:
            names = sorted(os.listdir(self._requests))
        except OSError:
            return
        for name in names:
            if name.startswith(".") or not name.endswith(".json") or not name.startswith(self._prefixes):
                continue
            source = self._requests / name
            request_id = name.split("--", 1)[1][:-5]
            if not self._owned_by_this_user(source):
                self.recorder.event("request_rejected", request_id=request_id, reason="not written by this user")
                continue
            if store.read_json(source) is None and self._younger_than(source, WRITING_SECONDS):
                continue  # a request an agent is still writing: the next look finds it whole
            self._claimed.mkdir(parents=True, exist_ok=True)
            claimed = self._claimed / name
            try:
                os.replace(source, claimed)  # a claim: if another process took it first, this fails
            except OSError:
                continue
            request = store.read_json(claimed)
            with contextlib.suppress(OSError):
                os.remove(claimed)
            self.handle(request_id, request)

    @staticmethod
    def _younger_than(path, seconds) -> bool:
        try:
            return time.time() - os.stat(path).st_mtime < seconds
        except OSError:
            return False

    @staticmethod
    def _owned_by_this_user(path) -> bool:
        if not hasattr(os, "getuid"):
            return True
        try:
            return os.stat(path).st_uid == os.getuid()
        except OSError:
            return False

    def handle(self, request_id, request) -> None:
        if not isinstance(request, dict) or "action" not in request:
            return self._respond(request_id, None, False, error="the request is not a JSON object with an 'action'")
        action = str(request["action"])
        self.recorder.event("request", request_id=request_id, action=action,
                            summary={key: value for key, value in request.items() if key not in ("id", "expr", "text")})
        handler = getattr(self, "do_" + action, None) if re.fullmatch(r"[a-z_]+", action) else None
        self._window_note = None
        if handler is None:
            known = sorted(name[3:] for name in dir(self) if name.startswith("do_"))
            return self._respond(request_id, action, False, error=f"unknown action {action!r}", actions=known)
        try:
            outcome = handler(request)
        except ActionError as error:
            return self._respond(request_id, action, False, error=str(error), **error.detail)
        except Exception as error:
            return self._respond(request_id, action, False, error=f"{type(error).__name__}: {error}")
        if isinstance(outcome, Deferred):
            pending = Pending(request_id, request, outcome, self._window_note)
            self._pending.append(pending)
            QtCore.QTimer.singleShot(0, lambda: self._run_pending(pending))
        else:
            self._respond(request_id, action, True, result=outcome, window_note=self._window_note)

    def _run_pending(self, pending: Pending) -> None:
        try:
            pending.result = pending.deferred.run()
        except ActionError as error:
            pending.error = (str(error), error.detail)
        except Exception as error:
            pending.error = (f"{type(error).__name__}: {error}", {})
        pending.finished_at = time.monotonic()

    def finish_pending(self) -> None:
        """Answer the actions that have run and settled, and those that have run too long (a modal dialog is open)."""
        now = time.monotonic()
        for pending in list(self._pending):
            action = pending.request["action"]
            done = pending.finished_at is not None
            settled = done and now - pending.finished_at >= pending.deferred.settle_ms / 1000
            if settled:
                self._pending.remove(pending)
                if pending.error is not None:
                    message, detail = pending.error
                    if not pending.responded:
                        self._respond(pending.request_id, action, False, error=message, **detail)
                    continue
                result = pending.result
                try:
                    if pending.deferred.after is not None:
                        result = pending.deferred.after(result)
                except Exception as error:
                    result = {**(result or {}), "after_error": f"{type(error).__name__}: {error}"}
                if pending.responded:
                    self.recorder.event("action_completed", request_id=pending.request_id, action=action)
                else:
                    self._respond(pending.request_id, action, True, result=result, window_note=pending.window_note)
            elif not pending.responded and now >= pending.deadline:
                pending.responded = True
                self._respond(
                    pending.request_id, action, True, result={
                        "completed": False,
                        "note": "the action is still running, which usually means it opened a modal dialog; "
                                "use `windows` to see it, and capture it or click its buttons",
                        "windows": [self._brief(entry) for entry in self.probe.windows.inventory()],
                    })

    def _respond(self, request_id, action, ok, *, result=None, error=None, window_note=None, **detail) -> None:
        response = {"id": request_id, "ok": ok, "action": action, "pid": self.recorder.pid, "kind": self.recorder.kind,
                    "wall_ms": store.wall_ms()}
        if window_note:
            response["window_note"] = window_note
        if ok:
            response["result"] = result
        else:
            response["error"] = error
            if detail:
                response["detail"] = detail
        try:
            store.atomic_write_json(store.response_path(self.recorder.out, request_id), response, attempts=30)
        except OSError as problem:
            self.recorder.event("response_failed", request_id=request_id, error=str(problem))

    # --- helpers
    @staticmethod
    def _brief(entry) -> dict:
        return {key: entry[key] for key in ("index", "title", "class", "type", "active", "modal") if key in entry}

    def _window(self, request):
        self.probe.windows.scan()
        window, matches = targets.find_window(request.get("window"), serials=self.probe.windows.serials,
                                              include_hidden=bool(request.get("include_hidden")))
        if matches > 1:  # the answer says so: a close or a click must not go to a window by accident
            title = targets.window_title(window) or type(window).__name__
            self._window_note = (f"{matches} windows matched 'window'; used {title!r} (the active one, else the newest). "
                                 "Name it more exactly, for example a longer regular expression anchored with $.")
        return window, matches

    def _target(self, request, *, widget_key="widget"):
        """The (target, window) a request names: a widget of the window, or the window itself without a "widget"."""
        window, _matches = self._window(request)
        spec = request.get(widget_key)
        if spec is None:
            return targets.Target(window, window.rect().center(), "window"), window
        return targets.resolve_target(window, spec), window

    def _qtest(self):
        from PySide6 import QtTest  # imported when first needed: the program may not have it

        return QtTest.QTest

    @staticmethod
    def _modifiers(names) -> Qt.KeyboardModifier:
        combined = Qt.KeyboardModifier.NoModifier
        for name in (names or []):
            try:
                combined |= MODIFIERS[str(name).lower()]
            except KeyError:
                raise ActionError(f"unknown modifier {name!r}", modifiers=sorted(MODIFIERS)) from None
        return combined

    @staticmethod
    def _deferred(request, run, *, settle=SETTLE_MS, after=None) -> Deferred:
        """A Deferred whose settle time and blocking time the request may set ("settle_ms", "block_ms")."""
        return Deferred(run, settle_ms=int(request.get("settle_ms", settle)), after=after,
                        block_ms=int(request.get("block_ms", BLOCK_MS)))

    def _after_ui(self, result):
        """What the program looks like once it has reacted: the visible windows, and what has the focus."""
        self.probe.windows.scan()
        result = dict(result or {})
        result["windows_after"] = [self._brief(entry) for entry in self.probe.windows.inventory()]
        result["focus_after"] = targets.focus_description()
        return result

    def _reveal(self, widget) -> None:
        """Scroll the widget into view first, as a user would to reach it."""
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QtWidgets.QScrollArea) and parent.widget() is not None and parent.isAncestorOf(widget):
                parent.ensureWidgetVisible(widget)
                return
            parent = parent.parentWidget()

    def _send_mouse(self, widget, point, button, modifiers, double) -> dict:
        window = widget.window()
        handle = window.windowHandle()
        if handle is None:
            raise ActionError("the window has no native window yet (it has not been shown)")
        self._reveal(widget)
        position = widget.mapTo(window, point)
        qtest = self._qtest()
        qtest.mouseMove(handle, position)  # the pointer arrives first, as it does for a user
        (qtest.mouseDClick if double else qtest.mouseClick)(handle, button, modifiers, position)
        under = window.childAt(position) if window.isVisible() else None
        return {"at": [position.x(), position.y()], "window": targets.window_title(window) or type(window).__name__,
                "widget_under_pointer": ({"class": type(under).__name__, "objectName": under.objectName(),
                                          "text": targets.widget_text(under)[:60]} if under is not None else None),
                "target_under_pointer": under is widget or (under is not None and widget.isAncestorOf(under))}

    def _focus_target(self, request):
        """The widget keys go to: the one the request names, else the one with the keyboard focus."""
        if request.get("widget") is not None:
            return self._target(request)[0].widget
        focused = QtWidgets.QApplication.focusWidget()
        if targets.alive(focused):
            return focused
        window, _ = self._window(request)
        return window

    # --- the actions: reading
    def do_ping(self, request):
        app = QtWidgets.QApplication.instance()
        return {"pid": self.recorder.pid, "kind": self.recorder.kind, "platform": app.platformName(),
                "uptime_s": round((store.wall_ms() - self.recorder.boot_wall_ms) / 1000, 2),
                "windows": len(self.probe.windows.inventory()), "viewer": self.probe.viewer() is not None}

    def do_windows(self, request):
        self.probe.windows.scan()
        return {"windows": self.probe.windows.inventory(), "focus": targets.focus_description(),
                "step": self.recorder.steps.current()["label"]}

    def do_tree(self, request):
        window, matches = self._window(request)
        root = window
        if request.get("widget") is not None:
            root = targets.resolve_target(window, request["widget"]).widget
        return {"window": targets.window_title(window) or type(window).__name__, "matches": matches,
                "tree": targets.widget_tree(root, window=window, max_nodes=int(request.get("max_nodes", 3000)),
                                            only_visible=not request.get("include_hidden", False),
                                            depth=int(request.get("depth", -1)))}

    def do_capture(self, request):
        if request.get("all"):
            self.probe.windows.scan()
            shots = [self.probe.capture(widget, reason="request")
                     for widget in targets.top_level_windows() if not targets.is_popup_like(widget)
                     and not self.probe.is_sensitive(widget)]
            return {"captures": shots}
        target, window = self._target(request)
        if self.probe.is_sensitive(window) and not request.get("force"):
            raise ActionError("this window looks like it holds a key or password; add \"force\": true to capture it anyway")
        return self.probe.capture(target.widget, reason="request", title=request.get("name"))

    def do_clipboard(self, request):
        clipboard = QtGui.QGuiApplication.clipboard()
        if "set" in request:
            clipboard.setText(str(request["set"]))
        mime = clipboard.mimeData()
        return {"text": clipboard.text(), "formats": list(mime.formats()) if mime is not None else []}

    def do_dialog_answers(self, request):
        queue = self.probe.file_dialogs
        if "answers" in request or request.get("clear"):
            try:
                count = queue.push(request.get("answers", []), clear=bool(request.get("clear")))
            except ValueError as error:
                raise ActionError(str(error)) from None
        else:
            count = len(queue.pending())
        return {"queued": count, "pending": queue.pending()}

    def do_eval(self, request):
        """Evaluate an expression in the program (``mode: "exec"`` runs statements and returns ``result``). Test-only."""
        namespace = {"app": QtWidgets.QApplication.instance(), "QtCore": QtCore, "QtGui": QtGui, "QtWidgets": QtWidgets,
                     "sys": sys, "main": sys.modules.get("__main__"), "viewer": self.probe.viewer(),
                     "windows": targets.top_level_windows(), "targets": targets, "result": None}
        if request.get("window") is not None or request.get("widget") is not None:
            target, window = self._target(request)
            namespace.update(window=window, widget=target.widget)
        source = str(request.get("expr", ""))
        if request.get("mode") == "exec":
            exec(compile(source, "<probe eval>", "exec"), namespace)  # noqa: S102 (a test tool, on purpose)
            value = namespace.get("result")
        else:
            value = eval(compile(source, "<probe eval>", "eval"), namespace)  # noqa: S307
        return {"value": json_safe(value), "repr": repr(value)[:2000]}

    # --- the actions: acting
    def do_click(self, request):
        target, window = self._target(request)
        button = BUTTONS.get(str(request.get("button", "left")).lower())
        if button is None:
            raise ActionError("button is left, right or middle")
        modifiers = self._modifiers(request.get("modifiers"))
        double = bool(request.get("double"))
        link = request.get("link")
        enabled = target.widget.isEnabled()

        def run():
            point = target.point
            if link is not None:
                point = self._link_point(target.widget, link)
            result = self._send_mouse(target.widget, point, button, modifiers, double)
            return {**result, "widget": target.description, "enabled": enabled,
                    **({} if enabled else {"warning": "the widget is disabled; a click does nothing"})}

        return self._deferred(request, run, after=self._after_ui)

    def do_hover(self, request):
        target, _window = self._target(request)

        def run():
            widget = target.widget
            handle = widget.window().windowHandle()
            if handle is None:
                raise ActionError("the window has no native window yet")
            self._reveal(widget)
            position = widget.mapTo(widget.window(), target.point)
            self._qtest().mouseMove(handle, position)
            return {"at": [position.x(), position.y()], "widget": target.description}

        return self._deferred(request, run, settle=900, after=self._after_ui)

    def _link_point(self, label, wanted):
        """A point on a link of a label, found by estimate and confirmed with the label's own linkHovered."""
        if not isinstance(label, QtWidgets.QLabel):
            raise ActionError("'link' needs a QLabel", widget=type(label).__name__)
        handle = label.window().windowHandle()
        if handle is None:
            raise ActionError("the window has no native window yet")
        hovered = []

        def note(link):
            hovered.append(link)

        label.linkHovered.connect(note)
        try:
            for point in targets.link_points(label, str(wanted)):
                hovered.clear()
                self._qtest().mouseMove(handle, label.mapTo(label.window(), point))
                if hovered and hovered[-1]:
                    return point
        finally:
            label.linkHovered.disconnect(note)
        raise ActionError("no link in the label matches", link=str(wanted), label_text=label.text()[:200])

    def do_type(self, request):
        text = str(request.get("text", ""))
        clear = bool(request.get("clear"))
        explicit = request.get("widget") is not None
        target = self._target(request)[0] if explicit else None
        if not explicit and not targets.alive(QtWidgets.QApplication.focusWidget()):
            raise ActionError("nothing has the keyboard focus; name a widget")

        def run():
            widget = target.widget if target is not None else QtWidgets.QApplication.focusWidget()
            if explicit and not widget.hasFocus():  # as a user does: click the field first
                self._send_mouse(widget, target.point, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, False)
                focused = QtWidgets.QApplication.focusWidget()
                widget = focused if targets.alive(focused) else widget
            qtest = self._qtest()
            if clear:
                qtest.keyClick(widget, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
                qtest.keyClick(widget, Qt.Key.Key_Delete)
            self._send_text(widget, text)
            return {"typed": len(text), "into": f"{type(widget).__name__} {widget.objectName()}".strip()}

        return self._deferred(request, run, after=self._after_ui)

    def _send_text(self, widget, text) -> None:
        """Type ``text`` into ``widget``: QTest for plain ASCII (its character conversion crashes on anything else), key events otherwise."""
        qtest = self._qtest()
        chunk = []

        def flush():
            if chunk:
                qtest.keyClicks(widget, "".join(chunk))
                chunk.clear()

        for character in text:
            if character == "\n":
                flush()
                qtest.keyClick(widget, Qt.Key.Key_Return)
            elif 32 <= ord(character) < 127:
                chunk.append(character)
            else:
                flush()
                for kind in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease):
                    QtWidgets.QApplication.sendEvent(widget, QtGui.QKeyEvent(kind, 0, Qt.KeyboardModifier.NoModifier, character))
        flush()

    def do_key(self, request):
        keys = request.get("keys", request.get("key"))
        keys = [keys] if isinstance(keys, str) else list(keys or [])
        if not keys:
            raise ActionError("'keys' is a key such as \"Return\" or \"Ctrl+A\", or a list of them")
        combinations = []
        for name in keys:
            sequence = QtGui.QKeySequence(str(name))
            if sequence.count() != 1 or sequence[0].key() == Qt.Key.Key_unknown:
                raise ActionError(f"cannot read the key {name!r}", hint="names as in QKeySequence: Return, Esc, Tab, Ctrl+A, F5")
            combinations.append(sequence[0])
        widget = self._focus_target(request)

        def run():
            target = widget if targets.alive(widget) else QtWidgets.QApplication.focusWidget()
            for combination in combinations:
                self._qtest().keyClick(target, combination.key(), combination.keyboardModifiers())
            return {"keys": [str(name) for name in keys], "to": f"{type(target).__name__} {target.objectName()}".strip()}

        return self._deferred(request, run, after=self._after_ui)

    def do_set_value(self, request):
        if "value" not in request:
            raise ActionError("'value' is missing")
        target, _window = self._target(request)
        value = request["value"]

        def run():
            return {"widget": target.description, "value": self._set_value(target.widget, value)}

        return self._deferred(request, run, after=self._after_ui)

    def _set_value(self, widget, value):
        """Give a widget a value the way its own signals expect: a combo and a field also say a person chose it."""
        if isinstance(widget, QtWidgets.QComboBox):
            index = value if isinstance(value, int) and not isinstance(value, bool) else widget.findText(
                str(value), Qt.MatchFlag.MatchFixedString)
            if not 0 <= index < widget.count():
                raise ActionError(f"the combo box has no entry {value!r}", items=[widget.itemText(i) for i in range(widget.count())])
            widget.setCurrentIndex(index)
            widget.activated.emit(index)  # what choosing it in the popup would add to currentIndexChanged
            return widget.currentText()
        if isinstance(widget, QtWidgets.QAbstractButton):
            if not widget.isCheckable():
                raise ActionError("the button is not checkable; use click")
            wanted = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes")
            if widget.isChecked() != wanted:
                widget.click()  # a toggle switch and a check box change as they do for a user's click
            return widget.isChecked()
        if isinstance(widget, QtWidgets.QDoubleSpinBox):
            widget.setValue(float(value))
            return widget.value()
        if isinstance(widget, QtWidgets.QSpinBox):
            widget.setValue(int(value))
            return widget.value()
        if isinstance(widget, (QtWidgets.QSlider, QtWidgets.QDial, QtWidgets.QScrollBar)):
            widget.setValue(int(value))
            return widget.value()
        if isinstance(widget, QtWidgets.QLineEdit):
            widget.setText(str(value))
            widget.textEdited.emit(str(value))
            return widget.text() if not targets.is_secret_field(widget) else "<hidden>"
        if isinstance(widget, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            widget.setPlainText(str(value))
            return widget.toPlainText()[:200]
        if isinstance(widget, QtWidgets.QTabBar):
            widget.setCurrentIndex(int(value))
            return widget.currentIndex()
        raise ActionError(f"set_value does not know a {type(widget).__name__}; use click, type or eval")

    def do_close(self, request):
        window, _matches = self._window(request)
        title = targets.window_title(window) or type(window).__name__

        def run():
            closed = window.close()  # sends the close event the title bar's button sends; False: the window refused
            return {"window": title, "accepted": bool(closed)}

        return self._deferred(request, run, after=self._after_ui)

    def do_resize(self, request):
        window, _matches = self._window(request)
        before = window.size()
        if request.get("to_minimum"):
            size = QtCore.QSize(1, 1)  # Qt raises it to the smallest the layout allows, as dragging an edge would
        elif "width" in request and "height" in request:
            size = QtCore.QSize(int(request["width"]), int(request["height"]))
        else:
            raise ActionError("give 'width' and 'height', or 'to_minimum': true")

        def run():
            window.resize(size)
            return {"from": [before.width(), before.height()]}

        def after(result):
            now = window.size()
            minimum = window.minimumSize()
            return {**result, "size": [now.width(), now.height()], "min_size": [minimum.width(), minimum.height()],
                    "min_size_hint": [window.minimumSizeHint().width(), window.minimumSizeHint().height()],
                    **self._after_ui({})}

        return self._deferred(request, run, settle=250, after=after)

    def do_viewer_command(self, request):
        """Run a command line in the Viewer as if it were typed into its command line (a Viewer process only)."""
        command = str(request.get("command", "")).strip()
        viewer = self.probe.viewer()
        if not command:
            raise ActionError("'command' is empty")
        if viewer is None:
            raise ActionError(f"this process has no Viewer (it is the {self.recorder.kind})")
        outcome_holder = {}
        printed_from = self._printed_mark()

        def run():
            engine = sys.modules.get("Command_Engine")
            recording = engine.recorded_outcome() if engine is not None and hasattr(engine, "recorded_outcome") \
                else contextlib.nullcontext({"status": None})
            with recording as outcome:
                viewer.process_command(command)  # the call the command line's Enter makes
            outcome_holder["status"] = outcome["status"]
            return {"command": command}

        def after(result):
            status_text = getattr(getattr(viewer, "console_text", None), "text", "")
            return {**result, "outcome": outcome_holder.get("status"),
                    "queued_or_pending": outcome_holder.get("status") is None,
                    "status_text": str(status_text), "printed": self._printed_since(printed_from),
                    **self._after_ui({})}

        return self._deferred(request, run, settle=VIEWER_SETTLE_MS, after=after)

    @staticmethod
    def _printed_mark():
        capture = sys.modules.get("utilities.Console_Capture")
        current = capture.current() if capture is not None else None
        lines = current.snapshot() if current is not None else []
        return lines[-1].seq if lines else -1

    @staticmethod
    def _printed_since(mark) -> list:
        capture = sys.modules.get("utilities.Console_Capture")
        current = capture.current() if capture is not None else None
        if current is None:
            return []
        return [line.text for line in current.snapshot() if line.seq > mark][-PRINTED_LINES:]
