# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Pictures of windows and widgets, made inside the process with QWidget.grab().

grab() renders the widget with Qt's own painting, so it needs nothing from the desktop: no screenshot
permission, no compositor, the same on Wayland, XWayland, X11, Windows and macOS. The picture has the widget's
device pixel ratio (the screen's scale). A widget that is drawn by something other than Qt's painter (an
OpenGL canvas, a web view) may come out blank or black: every capture says where such widgets are in it and
whether that part of the picture is one flat colour, so a reader knows what to trust.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui

from emapssn_desktop_probe import store, targets

# Widgets drawn by something other than Qt's painter, by class name anywhere in the class's bases.
EXTERNALLY_DRAWN = ("QOpenGLWidget", "QQuickWidget", "QWebEngineView", "QGLWidget")
SAMPLE_GRID = 8


def flat_colour(image: QtGui.QImage, rect: QtCore.QRect):
    """The colour (#rrggbb) of a part of a picture that has just one, else None; sampled on a grid, so it is quick."""
    rect = rect.intersected(image.rect())
    if rect.isEmpty():
        return None
    first = None
    for row in range(SAMPLE_GRID):
        for column in range(SAMPLE_GRID):
            x = rect.x() + (rect.width() - 1) * column // max(SAMPLE_GRID - 1, 1)
            y = rect.y() + (rect.height() - 1) * row // max(SAMPLE_GRID - 1, 1)
            colour = image.pixelColor(x, y).name()
            if first is None:
                first = colour
            elif colour != first:
                return None
    return first


def drawn_elsewhere(widget, image: QtGui.QImage, scale: float) -> list[dict]:
    """The parts of the picture of ``widget`` that Qt's painter does not draw, and whether each came out flat."""
    found = []
    for _path, child in targets.walk(widget):
        names = [base.__name__ for base in type(child).__mro__]
        match = next((name for name in EXTERNALLY_DRAWN if name in names), None)
        if match is None or not child.isVisible():
            continue
        top_left = child.mapTo(widget, QtCore.QPoint(0, 0))
        rect = QtCore.QRect(int(top_left.x() * scale), int(top_left.y() * scale),
                            int(child.width() * scale), int(child.height() * scale))
        found.append({"class": type(child).__name__, "kind": match, "objectName": child.objectName() or "",
                      "rect": [top_left.x(), top_left.y(), child.width(), child.height()],
                      "flat_colour": flat_colour(image, rect)})
    return found


def file_name(kind, pid, sequence, step_label, title) -> str:
    return f"{kind}-{pid}-{sequence:04d}-{store.slug(step_label, 30)}-{store.slug(title, 40)}.png"


def capture_widget(probe, widget, *, reason: str, title: str | None = None) -> dict:
    """Grab ``widget``, save it under captures/, note it in the event log, and describe it.

    ``probe`` supplies the recorder (kind, pid, out, the current step) and the capture counter.
    """
    recorder = probe.recorder
    window = widget.window()
    window_name = targets.window_title(window) or type(window).__name__
    label = title or (window_name if widget is window else f"{window_name} {type(widget).__name__}"
                      f"{('-' + widget.objectName()) if widget.objectName() else ''}")
    pixmap = widget.grab()
    if pixmap.isNull():
        raise targets.ActionError("the widget could not be grabbed (it has no size, or is not drawable)")
    sequence = probe.next_capture_number()
    step = recorder.steps.current()
    name = file_name(recorder.kind, recorder.pid, sequence, step["label"], label)
    path = Path(recorder.out) / "captures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(path), "PNG"):
        raise targets.ActionError(f"the picture could not be saved to {path}")
    scale = pixmap.devicePixelRatio() or 1.0
    image = pixmap.toImage()
    description = {
        "file": f"captures/{name}", "path": str(path), "width": pixmap.width(), "height": pixmap.height(),
        "dpr": round(scale, 4), "logical": [widget.width(), widget.height()], "title": label,
        "window": window_name, "reason": reason, "step": step["label"], "step_seq": step["seq"], "number": sequence,
        "flat_colour": flat_colour(image, image.rect()), "drawn_elsewhere": drawn_elsewhere(widget, image, scale),
        "screen": targets.screen_description(window),
    }
    recorder.event("capture", **description)
    return description
