# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Console window: what this program's process has printed, in a window of the Studio theme.

Config, Tools, VR Config and the Viewer no longer show a terminal on Windows,
so everything they print is kept (utilities.Console_Capture). The Viewer, which
runs for hours and prints the output of its commands, shows it here: live, in
the order it arrived, with the time of each line. It shows the last
Console_Capture.RING_LINES lines; the log file the footer names holds all of
them. Config and Tools show what they print on the start-up splash (desktop/Splash.py,
Show details) and keep it in temp/logs.

* A line is an error, a warning or plain output, decided by what it says
  (classify), not by the stream it came on: Python, Qt and CUDA libraries write
  their progress and notices to stderr. Errors are in the theme's danger colour
  and warnings in its warning colour. The lines that follow an error or
  warning and are indented (a traceback's frames, a warning's source line)
  take its colour too.
* The search field highlights every match and counts them; Enter and the
  arrows, or F3 and Shift+F3, go from one to the next. The filter shows all
  output, only warnings and errors, or only errors. Follow output keeps the
  newest line in view, and turns itself off when the view is scrolled up and
  on again at the bottom.
* Copy puts the selection (or everything shown) on the clipboard, Save writes
  it to a file, Clear view empties the window (the log file keeps everything),
  and the last two buttons open the log file and show its folder.

A progress bar that rewrites its line with carriage returns is one line that
changes where it is. The window feeds from the capture only while it is shown.

show_console opens this process's Console (or brings it forward): the Viewer's
`console` command, its sidebar button and its Session card's link call it.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import time
from pathlib import Path

import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QKeySequence,
    QShortcut,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.Desktop_App import (
    ResponsiveFlowLayout,
    SteadyPopupComboBox,
    ToggleSwitch,
    open_in_file_manager,
    qt_monospace_font,
)
from desktop.Studio_Theme import (
    QSS_ICON_VARIANTS,
    TOKENS,
    coloured_svg,
    icon_svg,
    page_tokens,
    qss_icon_path,
    set_role,
    studio_icon,
    studio_palette,
    studio_stylesheet,
    theme_color,
    theme_tokens,
)
from utilities import Console_Capture, Desktop_Open
from utilities.Line_Kinds import (  # noqa: F401  (re-exported: the window's callers and tests name them here)
    ERROR, ERROR_PATTERN, INFO, WARNING, WARNING_PATTERN, Classifier, classify,
)

# What makes a line an error or a warning is utilities/Line_Kinds.py (the start-up splash colours by it too).
# The colour behind the search's matches: a pale amber on the light page, a dark one on the dark.
MATCH_LIGHT = "#fde68a"
MATCH_DARK = "#854d0e"
# How many matches the search counts and highlights.
MAX_MATCHES = 5000
# The window collects what arrives and draws it this often (ms), so a burst of 100,000 lines
# is a few repaints.
FLUSH_MS = 50


def dark_tokens(color=None):
    """The window's tokens in a dark page: the neutral and accent shades of the pages' dark mode."""
    color = color or theme_color()
    tokens = theme_tokens(color)
    tokens.update(page_tokens(color, dark=True))
    tokens.update({
        "field": tokens["surface"], "field_disabled": tokens["window"],
        "danger": "#f87171", "warning": "#fbbf24", "success": "#4ade80",
        "tooltip": "#3f3f46", "tooltip_text": "#fafafa",
        "track": "#27272a", "tint": "#1f1f23", "tint_hover": "#2a2a30", "separator": "#3f3f46",
        "histogram": "#2a2a30", "accent_text": tokens["accent"],
    })
    return tokens


_dark_icon_folder = None


def _dark_stylesheet(tokens):
    """The Studio stylesheet in the dark tokens, with its arrows in the dark colours.

    The stylesheet's arrows are pre-coloured files for the theme's own light tokens (icons/qss); the dark
    tokens' colours have none, so they are written, once, into a folder of this process's own.
    """
    global _dark_icon_folder
    sheet = studio_stylesheet(tokens)
    for name, token in QSS_ICON_VARIANTS:
        path = qss_icon_path(name, token, tokens)
        if path.is_file():
            continue
        if _dark_icon_folder is None:
            _dark_icon_folder = Path(tempfile.mkdtemp(prefix="ssn_console_icons_"))
            atexit.register(shutil.rmtree, _dark_icon_folder, ignore_errors=True)  # nothing stays in TEMP
        copy = _dark_icon_folder / path.name
        if not copy.is_file():
            copy.write_text(coloured_svg(icon_svg(name), tokens[token]), encoding="utf-8")
        sheet = sheet.replace(path.as_posix(), copy.as_posix())
    return sheet


class _ElidedLabel(QLabel):
    """A label that shows as much of its text as fits, cut in the middle, with the whole in its tooltip."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ""
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

    def setFullText(self, text):
        self._full = text
        self.setToolTip(text)
        self._elide()

    def _elide(self):
        shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, max(0, self.width()))
        if shown != self.text():
            self.setText(shown)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def minimumSizeHint(self):
        return QSize(40, super().minimumSizeHint().height())


class ConsoleWindow(QWidget):
    """The Console of one process. ``capture`` is its utilities.Console_Capture.Recorder."""

    def __init__(self, capture, parent=None, *, dark=False, title=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.capture = capture
        self.dark = dark
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowTitle(title or QCoreApplication.translate("Console", "Console"))
        self.setObjectName("consoleWindow")
        self.resize(960, 600)
        self.setMinimumSize(520, 320)
        self._pending = []
        self._filter = "all"
        self._floor_seq = 0          # Clear view hides every line up to this one
        self._last_seq = None        # the line the last block shows, which its next state replaces
        self._classifier = Classifier()
        self._live = False
        self._following = True
        self._setting_scroll = False
        self._match_cursor = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        bar = ResponsiveFlowLayout(spacing=6)
        self.search = QLineEdit()
        self.search.setPlaceholderText(QCoreApplication.translate("Console", "Search the output…"))
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumWidth(180)
        self.search.setAccessibleName(QCoreApplication.translate("Console", "Search the output…"))
        self.previous_button = self._button("chevron-up", QCoreApplication.translate("Console", "Previous match (Shift+F3)"))
        self.next_button = self._button("chevron-down", QCoreApplication.translate("Console", "Next match (F3)"))
        self.match_label = QLabel("")
        set_role(self.match_label, "subtle")
        self.filter_box = SteadyPopupComboBox()
        self.filter_box.setSizeAdjustPolicy(SteadyPopupComboBox.SizeAdjustPolicy.AdjustToContents)
        for label, key in (
            (QCoreApplication.translate("Console", "All output"), "all"),
            (QCoreApplication.translate("Console", "Warnings and errors"), WARNING),
            (QCoreApplication.translate("Console", "Errors only"), ERROR),
        ):
            self.filter_box.addItem(label, key)
        self.follow_switch = ToggleSwitch(QCoreApplication.translate("Console", "Follow output"), QCoreApplication.translate("Console", "Follow output"))
        self.follow_switch.setChecked(True)
        self.copy_button = self._button("copy", QCoreApplication.translate("Console", "Copy the selection, or all the output shown"))
        self.save_button = self._button("download", QCoreApplication.translate("Console", "Save the output shown to a file…"))
        self.clear_button = self._button("trash-2", QCoreApplication.translate("Console", "Clear the view (the log file keeps everything)"))
        self.log_button = self._button("file-text", QCoreApplication.translate("Console", "Open the log file"))
        self.folder_button = self._button("folder-open", QCoreApplication.translate("Console", "Show the log folder"))
        for widget in (self.previous_button, self.next_button, self.copy_button, self.save_button,
                       self.clear_button, self.log_button, self.folder_button):
            widget.setFixedWidth(36)
        bar.addWidget(self.search, 1)
        for widget in (self.previous_button, self.next_button, self.match_label, self.filter_box,
                       self.follow_switch, self.copy_button, self.save_button, self.clear_button,
                       self.log_button, self.folder_button):
            bar.addWidget(widget)
        root.addLayout(bar)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setMaximumBlockCount(int(capture.lines.maxlen or Console_Capture.RING_LINES))
        self.text.setFont(qt_monospace_font(self.text.font()))
        self.text.setObjectName("consoleText")
        self.text.setAccessibleName(QCoreApplication.translate("Console", "Console"))
        root.addWidget(self.text, 1)

        footer = QHBoxLayout()
        self.count_label = QLabel("")
        set_role(self.count_label, "subtle")
        self.log_label = _ElidedLabel()
        set_role(self.log_label, "subtle")
        footer.addWidget(self.count_label)
        footer.addSpacing(12)
        footer.addWidget(self.log_label, 1)
        root.addLayout(footer)

        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.focus_search)
        QShortcut(QKeySequence("F3"), self, activated=lambda: self.find(True))
        QShortcut(QKeySequence("Shift+F3"), self, activated=lambda: self.find(False))

        self.search.textChanged.connect(self._search_changed)
        self.search.returnPressed.connect(lambda: self.find(True))
        self.previous_button.clicked.connect(lambda: self.find(False))
        self.next_button.clicked.connect(lambda: self.find(True))
        self.filter_box.currentIndexChanged.connect(self._filter_changed)
        self.follow_switch.toggled.connect(self._set_following)
        self.copy_button.clicked.connect(self.copy_text)
        self.save_button.clicked.connect(self.save_text)
        self.clear_button.clicked.connect(self.clear_view)
        self.log_button.clicked.connect(self.open_log)
        self.folder_button.clicked.connect(self.show_log_folder)
        self.text.verticalScrollBar().actionTriggered.connect(lambda _action: self._scrolled_by_user())
        self.text.verticalScrollBar().rangeChanged.connect(self._range_changed)
        self.text.viewport().installEventFilter(self)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._flush)
        self._listener = self._queue_line
        self._apply_theme()
        self._update_footer()

    # --- looks
    def _tokens(self):
        return dark_tokens() if self.dark else TOKENS

    def _button(self, icon, tooltip):
        button = QPushButton()
        button.setIcon(studio_icon(icon, getattr(self, "_icon_color", None)))
        button.setIconSize(QSize(16, 16))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setProperty("iconName", icon)
        set_role(button, "icon")
        return button

    def _apply_theme(self):
        """Take the colours of the current tokens: the text area's style, and the formats lines are drawn in."""
        self._restyling = True  # the style changes below are not news to changeEvent
        try:
            self._restyle(self._tokens())
        finally:
            self._restyling = False

    def _restyle(self, tokens):
        if self.dark:
            self._icon_color = tokens["text_muted"]
            self.setStyleSheet(_dark_stylesheet(tokens))
            self.setPalette(studio_palette(tokens))
            for button in self.findChildren(QPushButton):
                name = button.property("iconName")
                if name:
                    button.setIcon(studio_icon(name, tokens["text_muted"], tokens["text_disabled"]))
        self.text.setStyleSheet(
            f"QPlainTextEdit {{ background: {tokens['surface']}; color: {tokens['text_soft']};"
            f" border: 1px solid {tokens['border']}; border-radius: {tokens['radius_card']}px;"
            f" padding: 8px 6px 8px 10px; selection-background-color: {tokens['accent_soft']};"
            f" selection-color: {tokens['text']}; }}"
        )
        self.formats = {
            INFO: self._format(tokens["text_soft"]),
            WARNING: self._format(tokens["warning"]),
            ERROR: self._format(tokens["danger"]),
            "time": self._format(tokens["text_subtle"]),
        }
        self._highlight_color = QColor(MATCH_DARK if self.dark else MATCH_LIGHT)

    @staticmethod
    def _format(color):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        return fmt

    def changeEvent(self, event):
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange) and hasattr(self, "text") \
                and not getattr(self, "_restyling", False):
            self._apply_theme()
            if self.isVisible():
                self._render_all()
        super().changeEvent(event)

    # --- feeding
    def showEvent(self, event):
        super().showEvent(event)
        if not self._live:
            self._live = True
            self.capture.subscribe(self._listener)
            self.timer.start(FLUSH_MS)
        self._render_all()
        self._update_log_buttons()

    def hideEvent(self, event):
        super().hideEvent(event)
        if self._live:
            self._live = False
            self.capture.unsubscribe(self._listener)
            self.timer.stop()
            self._pending = []

    def _queue_line(self, line):
        self._pending.append(line)  # the reader thread only queues; the timer draws

    def _flush(self):
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        self._append(batch)

    def _render_all(self):
        self._pending = []
        self.text.clear()
        self._last_seq = None
        self._classifier = Classifier()
        self._append(self.capture.snapshot())

    def _wanted(self, kind):
        return not (self._filter == WARNING and kind == INFO or self._filter == ERROR and kind != ERROR)

    def _append(self, lines):
        scroll = self.text.verticalScrollBar()
        kept = scroll.value()
        document = self.text.document()
        cursor = QTextCursor(document)
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.beginEditBlock()
        for line in lines:
            if line.seq <= self._floor_seq:
                continue
            kind = self._classifier.kind(line.text, remember=not line.partial)
            if not self._wanted(kind):
                continue
            stamp = time.strftime("%H:%M:%S", time.localtime(line.time)) + "  "
            if line.seq == self._last_seq:  # the next state of a progress line replaces it
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
            elif document.characterCount() > 1:
                cursor.insertBlock()
            cursor.insertText(stamp, self.formats["time"])
            cursor.insertText(line.text, self.formats[kind])
            self._last_seq = line.seq
        cursor.endEditBlock()
        self._setting_scroll = True
        scroll.setValue(scroll.maximum() if self._following else kept)
        self._setting_scroll = False
        if self.search.text():
            self._highlight_matches()
        self._update_footer()

    def _update_footer(self):
        shown = self.text.blockCount() if self.text.document().characterCount() > 1 else 0
        written = self.capture.total
        text = QCoreApplication.translate("Console", "{shown} lines shown · {written} written since start-up").format(
            shown=f"{shown:,}", written=f"{written:,}"
        )
        if self.capture.dropped:
            text += " · " + QCoreApplication.translate("Console", "older lines are in the log file")
        self.count_label.setText(text)
        path = getattr(self.capture, "log_path", None)
        if path:
            self.log_label.setFullText(QCoreApplication.translate("Console", "Log: {path}").format(path=path))
        else:
            error = getattr(self.capture, "log_error", None) or ""
            self.log_label.setFullText(QCoreApplication.translate("Console", "No log file ({error})").format(error=error) if error else "")

    def _update_log_buttons(self):
        has_log = bool(getattr(self.capture, "log_path", None))
        self.log_button.setEnabled(has_log)
        self.folder_button.setEnabled(has_log)

    # --- scrolling
    def _set_following(self, on):
        self._following = bool(on)
        if on:
            self._setting_scroll = True
            scroll = self.text.verticalScrollBar()
            scroll.setValue(scroll.maximum())
            self._setting_scroll = False

    def _range_changed(self, _minimum, maximum):
        """The text grew, or the window changed shape: while following, stay at the end."""
        if self._following and not self._setting_scroll:
            self._setting_scroll = True
            self.text.verticalScrollBar().setValue(maximum)
            self._setting_scroll = False

    def _scrolled_by_user(self):
        QTimer.singleShot(0, self._follow_from_scroll)

    def _follow_from_scroll(self):
        scroll = self.text.verticalScrollBar()
        at_bottom = scroll.value() >= scroll.maximum() - 1
        if at_bottom != self._following:
            self.follow_switch.setChecked(at_bottom)

    def eventFilter(self, watched, event):
        if watched is self.text.viewport() and event.type() == QEvent.Type.Wheel:
            self._scrolled_by_user()
        return super().eventFilter(watched, event)

    # --- the filter and Clear view
    def _filter_changed(self, _index):
        self._filter = self.filter_box.currentData()
        self._render_all()

    def clear_view(self):
        snapshot = self.capture.snapshot()
        self._floor_seq = snapshot[-1].seq if snapshot else self._floor_seq
        self._render_all()

    # --- searching
    def focus_search(self):
        self.search.setFocus()
        self.search.selectAll()

    def _search_changed(self, _text):
        self._match_cursor = None
        self.find(True, from_start=True)

    def _highlight_matches(self):
        """Highlight every match (up to MAX_MATCHES) and say how many there are; returns the count."""
        needle = self.search.text()
        selections = []
        count = 0
        if needle:
            document = self.text.document()
            highlight = QTextCharFormat()
            highlight.setBackground(self._highlight_color)
            cursor = QTextCursor(document)
            while count < MAX_MATCHES:
                cursor = document.find(needle, cursor)
                if cursor.isNull():
                    break
                extra = QTextEdit.ExtraSelection()
                extra.cursor, extra.format = QTextCursor(cursor), highlight
                selections.append(extra)
                count += 1
        self.text.setExtraSelections(selections)
        if not needle:
            self.match_label.setText("")
        elif count >= MAX_MATCHES:
            self.match_label.setText(QCoreApplication.translate("Console", "{count} or more matches").format(count=f"{MAX_MATCHES:,}"))
        elif count == 0:
            self.match_label.setText(QCoreApplication.translate("Console", "No matches"))
        else:
            self.match_label.setText(QCoreApplication.translate("Console", "%n match(es)", None, count))
        return count

    def find(self, forward, from_start=False):
        """Go to the next (or previous) match; wraps round at either end."""
        count = self._highlight_matches()
        needle = self.search.text()
        if not needle or not count:
            return False
        document = self.text.document()
        start = QTextCursor(document) if from_start else self.text.textCursor()
        flags = QTextDocument.FindFlag(0) if forward else QTextDocument.FindFlag.FindBackward
        found = document.find(needle, start, flags)
        if found.isNull():
            wrap = QTextCursor(document)
            if not forward:
                wrap.movePosition(QTextCursor.MoveOperation.End)
            found = document.find(needle, wrap, flags)
        if found.isNull():
            return False
        self.text.setTextCursor(found)
        self.text.centerCursor()
        if self.follow_switch.isChecked():
            self.follow_switch.setChecked(False)
        return True

    # --- Copy, Save and the log file
    def shown_text(self):
        """The text of every line shown, with its time."""
        return self.text.toPlainText()

    def copy_text(self):
        cursor = self.text.textCursor()
        # A selection spans blocks, which Qt joins with U+2029 (and a non-breaking space for a tab).
        text = (
            cursor.selectedText().replace("\u2029", "\n").replace("\u00a0", " ")
            if cursor.hasSelection() else self.shown_text()
        )
        QGuiApplication.clipboard().setText(text)
        return text

    def save_text(self):
        suggested = f"{self.capture.app}-console-{time.strftime('%Y%m%d-%H%M%S')}.txt"
        directory = str(Path(self.capture.log_path).parent) if getattr(self.capture, "log_path", None) else ""
        path, _filter = QFileDialog.getSaveFileName(
            self, QCoreApplication.translate("Console", "Save the output"), str(Path(directory) / suggested) if directory else suggested,
            QCoreApplication.translate("Console", "Text files (*.txt);;All files (*)"),
        )
        if not path:
            return None
        try:
            Path(path).write_text(self.shown_text() + "\n", encoding="utf-8")
        except OSError as error:
            QMessageBox.critical(self, QCoreApplication.translate("Console", "Console"), QCoreApplication.translate("Console", "The output could not be saved:\n{error}").format(error=error))
            return None
        return path

    def open_log(self):
        path = getattr(self.capture, "log_path", None)
        if path and not Desktop_Open.open_path(path):
            open_in_file_manager(str(Path(path).parent))

    def show_log_folder(self):
        path = getattr(self.capture, "log_path", None)
        if path:
            open_in_file_manager(str(Path(path).parent))


# =====================================================================
# One Console for the process
# =====================================================================

_window: ConsoleWindow | None = None


def console_available() -> bool:
    """Whether this process keeps its output (it was not started with --console, headless or under MCP)."""
    return Console_Capture.current() is not None


def not_available_text() -> str:
    return QCoreApplication.translate("Console",
        "This program's output is not kept here: it was started so that the output goes to the terminal "
        "it was started from (--console), where it can be read."
    )


def show_console(parent=None, capture=None, *, dark=False):
    """Open this process's Console, or bring it forward; returns the window, or None.

    With no capture to show, a message says so (a program started with --console prints to its terminal).
    """
    global _window
    capture = capture or Console_Capture.current()
    if capture is None:
        QMessageBox.information(parent, QCoreApplication.translate("Console", "Console"), not_available_text())
        return None
    if _window is None or not shiboken6.isValid(_window) or _window.capture is not capture:
        title = QCoreApplication.translate("Console", "Console")
        if parent is not None and parent.window() is not None and parent.window().windowTitle():
            title = QCoreApplication.translate("Console", "{window} – Console").format(window=parent.window().windowTitle())
        _window = ConsoleWindow(capture, None, dark=dark, title=title)
        # Where it opens: beside the window it belongs to, on its screen.
        if parent is not None and parent.window() is not None and parent.window().isVisible():
            frame = parent.window().frameGeometry()
            _window.move(frame.left() + 40, frame.top() + 40)
    _window.show()
    if _window.isMinimized():
        _window.showNormal()
    _window.raise_()
    _window.activateWindow()
    return _window

