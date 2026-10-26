# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Finding windows and widgets the way a person points at them, and describing them.

A window is named by a regular expression for its title (or its class or objectName). A widget is named by
what its label says or what it is called: objectName, accessibleName, visible text or tooltip, alone or
together, or by its path (the child indices ``tree`` prints). Parts of a widget that are not widgets of their
own are named too: a tab of a tab bar (``tab``), a row of a list, tree or combo popup (``item``) and an entry
of a menu or menu bar (``action``). Each of these resolves to a Target: the widget to act on and the point in it
to click.

Only the GUI thread may call into this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from emapssn_desktop_probe import store

Qt = QtCore.Qt
SECRET_WORDS = re.compile(r"(?i)api|key|token|secret|passw|credential")
TEXT_LIMIT = 300
POPUP_TYPES = (Qt.WindowType.Popup, Qt.WindowType.ToolTip)


class ActionError(Exception):
    """A request that cannot be carried out; ``detail`` goes into the response beside the message."""

    def __init__(self, message, **detail):
        super().__init__(message)
        self.detail = detail


@dataclass
class Target:
    """The widget to act on, the point in it to click (widget coordinates), and what it was called."""

    widget: QtWidgets.QWidget
    point: QtCore.QPoint
    description: str


def cpp_key(obj) -> int:
    """The address of the C++ object behind a wrapper: the same object has the same key from scan to scan."""
    return shiboken6.getCppPointer(obj)[0]


def alive(obj) -> bool:
    return obj is not None and shiboken6.isValid(obj)


# ---------------------------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------------------------

def normalize(text) -> str:
    """Text as a person compares it: no accelerator ampersands, one space between words, no case."""
    text = re.sub(r"&(?!&)", "", str(text or "")).replace("&&", "&")
    return " ".join(text.split()).casefold()


def is_secret_field(widget) -> bool:
    """A line edit that holds a password or key: its text is never read out."""
    if not isinstance(widget, QtWidgets.QLineEdit):
        return False
    if widget.echoMode() != QtWidgets.QLineEdit.EchoMode.Normal:
        return True
    return bool(SECRET_WORDS.search(widget.objectName() or "") or SECRET_WORDS.search(widget.accessibleName() or ""))


def widget_text(widget) -> str:
    """The text a widget shows: its label, title, current choice or content (a secret field says "<hidden>")."""
    try:
        if isinstance(widget, QtWidgets.QLineEdit):
            text = "<hidden>" if is_secret_field(widget) else widget.text()
        elif isinstance(widget, (QtWidgets.QAbstractButton, QtWidgets.QLabel)):
            text = widget.text()
        elif isinstance(widget, QtWidgets.QGroupBox):
            text = widget.title()
        elif isinstance(widget, QtWidgets.QComboBox):
            text = widget.currentText()
        elif isinstance(widget, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            text = widget.toPlainText()
        elif isinstance(widget, QtWidgets.QAbstractSpinBox):
            text = widget.text()
        elif isinstance(widget, QtWidgets.QMenu):
            text = widget.title()
        elif isinstance(widget, QtWidgets.QStatusBar):
            text = widget.currentMessage()
        elif isinstance(widget, QtWidgets.QProgressBar):
            text = widget.text()
        elif isinstance(widget, QtWidgets.QTabBar):
            text = widget.tabText(widget.currentIndex()) if widget.count() else ""
        else:
            text = ""
    except Exception:
        text = ""
    text = str(text or "")
    return text if len(text) <= TEXT_LIMIT else text[:TEXT_LIMIT] + "..."


def window_title(widget) -> str:
    try:
        return widget.windowTitle() or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------------

def window_type_name(widget) -> str:
    kind = widget.windowType()
    for name in ("Popup", "ToolTip", "SplashScreen", "Dialog", "Sheet", "Tool", "Drawer", "Desktop", "SubWindow"):
        if kind == getattr(Qt.WindowType, name):
            return name
    return "Window"


def is_popup_like(widget) -> bool:
    """Menus, combo popups and tooltips: windows that come and go with a click and are not captured on their own."""
    return widget.windowType() in POPUP_TYPES


def top_level_windows(*, include_hidden=False) -> list:
    app = QtWidgets.QApplication.instance()
    if app is None:
        return []
    windows = []
    for widget in app.topLevelWidgets():
        try:
            if widget.isWindow() and (include_hidden or widget.isVisible()):
                windows.append(widget)
        except RuntimeError:  # deleted while listing
            continue
    return windows


def focus_description(app=None) -> dict | None:
    """The widget that has the keyboard focus: its class and objectName, its window, and whether it draws focus."""
    app = app or QtWidgets.QApplication.instance()
    focused = app.focusWidget() if app else None
    if not alive(focused):
        return None
    window = focused.window()
    return {
        "class": type(focused).__name__, "objectName": focused.objectName() or "", "text": widget_text(focused),
        "window": window_title(window) or type(window).__name__,
        # Qt draws a focus rectangle only after the keyboard moved the focus (Tab); the style has the last word.
        "draws_focus": bool(focused.hasFocus() and window.testAttribute(Qt.WidgetAttribute.WA_KeyboardFocusChange)),
        "focus_policy": focused.focusPolicy().name,
    }


def screen_description(widget) -> dict:
    try:
        screen = widget.screen()
        geometry = screen.geometry()
        return {"name": screen.name(), "scale": round(screen.devicePixelRatio(), 4),
                "geometry": [geometry.x(), geometry.y(), geometry.width(), geometry.height()],
                "logical_dpi": round(screen.logicalDotsPerInch(), 2)}
    except Exception:
        return {"name": None, "scale": None}


def describe_window(widget) -> dict:
    """Everything the inventory says about a top-level window (geometry and sizes in logical pixels)."""
    geometry = widget.geometry()
    frame = widget.frameGeometry()
    minimum = widget.minimumSize()
    hint = widget.minimumSizeHint()
    screen = screen_description(widget)
    return {
        "class": type(widget).__name__, "objectName": widget.objectName() or "", "title": window_title(widget),
        "type": window_type_name(widget), "visible": widget.isVisible(), "active": widget.isActiveWindow(),
        "minimized": widget.isMinimized(), "maximized": widget.isMaximized(), "fullscreen": widget.isFullScreen(),
        "modal": widget.isModal(), "geometry": [geometry.x(), geometry.y(), geometry.width(), geometry.height()],
        "frame": [frame.x(), frame.y(), frame.width(), frame.height()],
        "min_size": [minimum.width(), minimum.height()], "min_size_hint": [hint.width(), hint.height()],
        "dpr": round(widget.devicePixelRatioF(), 4), "screen": screen["name"], "screen_scale": screen["scale"],
        "offscreen_only": widget.testAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen),
    }


def find_window(spec, *, serials=None, include_hidden=False):
    """The window ``spec`` names: a title regex, an index into the inventory, or {"title", "class", "objectName"}.

    With no spec, the active window, else the newest visible one that is not a popup. When several windows
    match, the active one wins, then a modal one, then the newest (``serials`` maps a window to the order in which
    the probe first saw it); the second result is how many matched.
    """
    windows = top_level_windows(include_hidden=include_hidden)
    serials = serials or {}
    windows.sort(key=lambda widget: serials.get(cpp_key(widget), 0))
    if isinstance(spec, int) and not isinstance(spec, bool):
        if not 0 <= spec < len(windows):
            raise ActionError(f"no window with index {spec}", windows=[window_title(w) for w in windows])
        return windows[spec], 1
    if isinstance(spec, str):
        spec = {"title": spec}
    spec = spec or {}
    matches = windows
    if spec.get("title"):
        pattern = re.compile(spec["title"], re.IGNORECASE)
        matches = [w for w in matches if pattern.search(window_title(w) or "")
                   or (not window_title(w) and pattern.search(type(w).__name__))]
    if spec.get("class"):
        matches = [w for w in matches if type(w).__name__ == spec["class"]]
    if spec.get("objectName"):
        matches = [w for w in matches if w.objectName() == spec["objectName"]]
    if not spec:
        matches = [w for w in matches if not is_popup_like(w)] or matches
    if not matches:
        raise ActionError("no window matches", window=spec,
                          windows=[window_title(w) or type(w).__name__ for w in windows])
    for preferred in (lambda w: w.isActiveWindow(), lambda w: w.isModal()):
        chosen = [w for w in matches if preferred(w)]
        if chosen:
            return chosen[-1], len(matches)
    return matches[-1], len(matches)


# ---------------------------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------------------------

def child_widgets(widget) -> list:
    """The widget children of ``widget`` that are not windows of their own, in the order Qt keeps them."""
    return [child for child in widget.children() if isinstance(child, QtWidgets.QWidget) and not child.isWindow()]


def widget_by_path(window, path: str):
    node = window
    for part in str(path).split("/"):
        if part == "":
            continue
        try:
            node = child_widgets(node)[int(part)]
        except (ValueError, IndexError):
            raise ActionError(f"no widget at path {path!r}", at=part) from None
    return node


def walk(window):
    """(path, widget) for the window and every widget below it, parents before children."""
    stack = [("", window)]
    while stack:
        path, widget = stack.pop()
        yield path, widget
        children = child_widgets(widget)
        for index in range(len(children) - 1, -1, -1):
            stack.append((f"{path}/{index}" if path else str(index), children[index]))


def widget_row(path, widget, window) -> dict:
    """One line describing a widget, for the lists of candidates in an error."""
    return {"path": path, "class": type(widget).__name__, "objectName": widget.objectName() or "",
            "text": widget_text(widget)[:60], "toolTip": (widget.toolTip() or "")[:60]}


def _spec_fields(spec) -> dict:
    if isinstance(spec, str):
        return {"any": spec}
    if isinstance(spec, dict):
        return dict(spec)
    raise ActionError("widget must be a string or an object", widget=repr(spec)[:100])


def _field_matches(have, want, *, exact: bool) -> bool:
    have, want = normalize(have), normalize(want)
    if not want:
        return False
    return have == want if exact else want in have


def _widget_matches(widget, fields, *, exact: bool) -> bool:
    if "class" in fields and type(widget).__name__ != fields["class"] and not any(
            base.__name__ == fields["class"] for base in type(widget).__mro__):
        return False
    tests = []
    if "objectName" in fields:
        tests.append(widget.objectName() == fields["objectName"] if exact
                     else _field_matches(widget.objectName(), fields["objectName"], exact=False))
    if "accessibleName" in fields:
        tests.append(_field_matches(widget.accessibleName(), fields["accessibleName"], exact=exact))
    if "text" in fields:
        tests.append(_field_matches(widget_text(widget), fields["text"], exact=exact))
    if "toolTip" in fields:
        tests.append(_field_matches(widget.toolTip(), fields["toolTip"], exact=exact))
    if "any" in fields:
        wanted = fields["any"]
        tests.append(
            widget.objectName() == wanted
            or (exact and type(widget).__name__ == wanted)
            or _field_matches(widget.accessibleName(), wanted, exact=exact)
            or _field_matches(widget_text(widget), wanted, exact=exact)
            or _field_matches(widget.toolTip(), wanted, exact=exact)
        )
    if tests:
        return all(tests)
    return "class" in fields


def find_targets(window, spec) -> list[Target]:
    """Every target in ``window`` that ``spec`` names, in tree order; exact matches first, else substring ones."""
    fields = _spec_fields(spec)
    only_visible = fields.get("visible", True)
    if "path" in fields:
        widget = widget_by_path(window, fields["path"])
        return [Target(widget, widget.rect().center(), f"path {fields['path']}")]
    for key, finder in (("tab", _tab_targets), ("item", _item_targets), ("action", _action_targets)):
        if key in fields:
            return finder(window, fields, only_visible)
    for exact in (True, False):
        found = []
        for path, widget in walk(window):
            if only_visible and not widget.isVisible():
                continue
            if _widget_matches(widget, fields, exact=exact):
                found.append(Target(widget, widget.rect().center(), f"{type(widget).__name__} {path or '(window)'}"))
        if found:
            return found
        if not any(key in fields for key in ("objectName", "accessibleName", "text", "toolTip", "any")):
            break
    return []


def resolve_target(window, spec) -> Target:
    """The one target ``spec`` names in ``window``; an error that lists what is there when none or several match."""
    fields = _spec_fields(spec)
    found = find_targets(window, spec)
    index = fields.get("index")
    if not found:
        candidates = [widget_row(path, widget, window) for path, widget in walk(window)
                      if widget.isVisible() and (widget_text(widget) or widget.objectName() or widget.toolTip())]
        raise ActionError("no widget matches", widget=fields, window=window_title(window) or type(window).__name__,
                          visible_candidates=candidates[:40])
    if index is not None:
        if not 0 <= int(index) < len(found):
            raise ActionError(f"index {index} is out of range", matches=len(found))
        return found[int(index)]
    if len(found) > 1:
        raise ActionError("more than one widget matches; add 'index' or name it more exactly", widget=fields,
                          matches=[{"index": position, "description": target.description,
                                    "text": widget_text(target.widget)[:60], "objectName": target.widget.objectName()}
                                   for position, target in enumerate(found[:15])])
    return found[0]


def _tab_targets(window, fields, only_visible):
    found = []
    for _path, widget in walk(window):
        if isinstance(widget, QtWidgets.QTabBar) and (widget.isVisible() or not only_visible):
            for index in range(widget.count()):
                if _field_matches(widget.tabText(index), fields["tab"], exact=True) or _field_matches(
                        widget.tabText(index), fields["tab"], exact=False):
                    found.append(Target(widget, widget.tabRect(index).center(), f"tab {widget.tabText(index)!r}"))
    return found


def _item_targets(window, fields, only_visible):
    found = []
    for _path, widget in walk(window):
        if isinstance(widget, QtWidgets.QAbstractItemView) and (widget.isVisible() or not only_visible):
            for index in _model_indexes(widget):
                text = widget.model().data(index, Qt.ItemDataRole.DisplayRole)
                if _field_matches(text, fields["item"], exact=True) or _field_matches(text, fields["item"], exact=False):
                    widget.scrollTo(index)
                    rect = widget.visualRect(index)
                    found.append(Target(widget.viewport(), rect.center(), f"item {str(text)!r}"))
    return found


def _model_indexes(view, limit=2000):
    """The visible rows of a list, table or tree (first column), expanded tree rows included."""
    model = view.model()
    if model is None:
        return []
    indexes, stack = [], [QtCore.QModelIndex()]
    while stack and len(indexes) < limit:
        parent = stack.pop()
        for row in range(model.rowCount(parent)):
            index = model.index(row, 0, parent)
            indexes.append(index)
            expanded = not isinstance(view, QtWidgets.QTreeView) or view.isExpanded(index)
            if expanded and model.rowCount(index) > 0:
                stack.append(index)
    return indexes


def _action_targets(window, fields, only_visible):
    found = []
    for _path, widget in walk(window):
        if isinstance(widget, (QtWidgets.QMenu, QtWidgets.QMenuBar)) and (widget.isVisible() or not only_visible):
            for action in widget.actions():
                if _field_matches(action.text(), fields["action"], exact=True) or _field_matches(
                        action.text(), fields["action"], exact=False):
                    found.append(Target(widget, widget.actionGeometry(action).center(), f"action {action.text()!r}"))
    return found


# ---------------------------------------------------------------------------------------------
# The widget tree
# ---------------------------------------------------------------------------------------------

def tree_node(path, widget, window, budget, *, only_visible, depth_left):
    """The tree below ``widget`` as nested dicts; ``budget`` ([n]) counts the nodes left to print."""
    budget[0] -= 1
    try:
        top_left = widget.mapTo(window, QtCore.QPoint(0, 0))
    except Exception:
        top_left = QtCore.QPoint(0, 0)
    node = {
        "path": path, "class": type(widget).__name__, "objectName": widget.objectName() or "",
        "accessibleName": widget.accessibleName() or "", "text": widget_text(widget),
        "toolTip": widget.toolTip() or "", "geometry": [top_left.x(), top_left.y(), widget.width(), widget.height()],
        "visible": widget.isVisible(), "enabled": widget.isEnabled(), "focus": widget.hasFocus(),
    }
    if isinstance(widget, QtWidgets.QAbstractButton) and widget.isCheckable():
        node["checked"] = widget.isChecked()
    if isinstance(widget, QtWidgets.QComboBox):
        node["items"] = [widget.itemText(i) for i in range(min(widget.count(), 60))]
        node["current_index"] = widget.currentIndex()
    if isinstance(widget, QtWidgets.QTabBar):
        node["tabs"] = [widget.tabText(i) for i in range(widget.count())]
    if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        node["value"] = widget.value()
    if isinstance(widget, QtWidgets.QLabel) and widget.textFormat() != Qt.TextFormat.PlainText and "<a " in widget.text():
        node["links"] = re.findall(r"""href=["']([^"']*)["']""", widget.text())[:10]
    children = []
    if depth_left != 0:
        for index, child in enumerate(child_widgets(widget)):
            if only_visible and not child.isVisible():
                continue
            if budget[0] <= 0:
                node["truncated"] = True
                break
            children.append(tree_node(f"{path}/{index}" if path else str(index), child, window, budget,
                                      only_visible=only_visible, depth_left=depth_left - 1))
    if children:
        node["children"] = children
    return node


def widget_tree(root, *, window=None, max_nodes=3000, only_visible=True, depth=-1) -> dict:
    """The tree below ``root``; geometry is in the coordinates of ``window`` (the root's own window by default)."""
    return tree_node("", root, window or root.window(), [max_nodes], only_visible=only_visible, depth_left=depth)


# ---------------------------------------------------------------------------------------------
# A link in a label
# ---------------------------------------------------------------------------------------------

def link_points(label: QtWidgets.QLabel, wanted: str):
    """Points in ``label`` (its own coordinates) to try for the link whose text or address is ``wanted``.

    The label lays its text out with a text document, so one is built the way the label builds it and the
    link's characters are located in it. The estimate is then confirmed by moving the mouse there (the label
    reports linkHovered), because the label's own alignment rules are not public.
    """
    document = QtGui.QTextDocument()
    document.setDocumentMargin(0)
    document.setDefaultFont(label.font())
    options = document.defaultTextOption()
    options.setWrapMode(QtGui.QTextOption.WrapMode.WordWrap if label.wordWrap() else QtGui.QTextOption.WrapMode.ManualWrap)
    options.setAlignment(label.alignment() & Qt.AlignmentFlag.AlignHorizontal_Mask)
    document.setDefaultTextOption(options)
    if label.textFormat() == Qt.TextFormat.PlainText:
        document.setPlainText(label.text())
    else:
        document.setHtml(label.text())
    contents = label.contentsRect().adjusted(label.margin(), label.margin(), -label.margin(), -label.margin())
    document.setTextWidth(contents.width())
    layout = document.documentLayout()
    height = layout.documentSize().height()
    offset_y = 0.0
    if label.alignment() & Qt.AlignmentFlag.AlignVCenter:
        offset_y = max((contents.height() - height) / 2, 0)
    elif label.alignment() & Qt.AlignmentFlag.AlignBottom:
        offset_y = max(contents.height() - height, 0)
    wanted_norm = normalize(wanted)
    points = []
    block = document.begin()
    while block.isValid():
        layout.blockBoundingRect(block)  # lays the block out; its lines do not exist before
        text_layout = block.layout()
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            fmt = fragment.charFormat()
            if fmt.isAnchor() and (normalize(fragment.text()) == wanted_norm or fmt.anchorHref() == wanted
                                   or wanted_norm in normalize(fragment.text()) or not wanted_norm):
                start = fragment.position() - block.position()
                for position in (start + fragment.length() // 2, start, start + max(fragment.length() - 1, 0)):
                    line = text_layout.lineForTextPosition(position)
                    if not line.isValid():
                        continue
                    x = line.cursorToX(position)[0] + 1
                    y = layout.blockBoundingRect(block).top() + line.y() + line.height() / 2
                    points.append(QtCore.QPoint(int(contents.x() + x), int(contents.y() + offset_y + y)))
            iterator += 1
        block = block.next()
    return points
