# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Console window: what this program's process has printed, in a window of the Studio theme.

Config, Tools, VR Config and the Viewer no longer show a terminal on Windows,
so everything they print is kept (utilities.Console_Capture). The Viewer, which
runs for hours and prints the output of its commands, shows it here: live, in
the order it arrived, with the time of each line in a gutter beside the text.
It shows the last
Console_Capture.RING_LINES lines; the log file the footer names holds all of
them. Config and Tools show what they print on the start-up splash (desktop/Splash.py,
Show details) and keep it in temp/logs.

* A line is an error, a warning or plain output, decided by what it says
  (classify), not by the stream it came on: Python, Qt and CUDA libraries write
  their progress and notices to stderr. Errors are in the theme's danger colour
  and warnings in its warning colour. The lines that follow an error or
  warning and are indented (a traceback's frames, a warning's source line)
  take its colour too.
* A command's colours show: the output may hold ANSI colour sequences
  (utilities/Terminal_Colors.py), which the window reads instead of showing as
  text. The 16 colours are the page's own shades, and any colour is moved
  lighter or darker until it shows on the page. A line with colours of its own
  keeps them, and the colouring by what a line says applies only to lines
  without.
* The time of each line is drawn in a gutter, as line numbers are, and is no part of
  the text: selecting, Copy, Save and the search see the output alone. The
  Times switch hides the gutter, and the choice is kept (app_settings.json).
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
`console` command and its Console button, at the bottom of the Viewer, call it.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import time
from pathlib import Path

import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, QRect, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QKeySequence,
    QPainter,
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
from utilities.App_Settings import AppSettingsError, read_app_settings, save_app_setting
from utilities.Line_Kinds import (  # noqa: F401  (re-exported: the window's callers and tests name them here)
    ERROR, ERROR_PATTERN, INFO, WARNING, WARNING_PATTERN, Classifier, classify,
)
from utilities.Terminal_Colors import DEFAULT_STYLE, ESC, palette_color, parse_ansi, readable_color, resolve_color

# What makes a line an error or a warning is utilities/Line_Kinds.py (the start-up splash colours by it too).
# The colour behind the search's matches: a pale amber on the light page, a dark one on the dark.
MATCH_LIGHT = "#fde68a"
MATCH_DARK = "#854d0e"
# How many matches the search counts and highlights.
MAX_MATCHES = 5000
# Whether the gutter of times shows: kept in app_settings.json, as the other windows keep their theme
# colour and language. Shown unless the user turned it off.
SHOW_TIMES_SETTING = "CONSOLE_SHOW_TIMES"
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


def saved_show_times(settings=None):
    """Whether the user left the times shown (the default): the saved choice, or True for none or an unreadable file."""
    if settings is None:
        try:
            settings = read_app_settings()
        except AppSettingsError:
            return True
    value = settings.get(SHOW_TIMES_SETTING)
    return value if isinstance(value, bool) else True


def save_show_times(on):
    """Keep the choice for the next Console; False when it could not be kept (the file holds more than this and cannot be read)."""
    try:
        save_app_setting(SHOW_TIMES_SETTING, bool(on))
    except (AppSettingsError, OSError):
        return False
    return True


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


def seconds_of_day(when):
    """The second of the (local) day ``when`` falls in: what a line keeps as its time."""
    stamp = time.localtime(when)
    return stamp.tm_hour * 3600 + stamp.tm_min * 60 + stamp.tm_sec


def clock_text(seconds):
    """HH:MM:SS for a second of the day, or "" for a block that has none."""
    if seconds < 0:
        return ""
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class _TimeGutter(QWidget):
    """The strip beside the text where each line's time is drawn, as line numbers are in a code editor."""

    def __init__(self, editor):
        super().__init__(editor)
        self._editor = editor
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, event):
        self._editor.paint_times(self, event)

    def wheelEvent(self, event):
        QApplication.sendEvent(self._editor.viewport(), event)  # the strip scrolls the text it sits beside


class ConsoleText(QPlainTextEdit):
    """The text area, with a gutter on its left that draws the time of each line.

    A line's time is the user state of its block (the second of the day), so it moves with the block when
    lines are trimmed from the top, and it is no part of the document: selecting, Copy, Save and the
    search see the lines alone. ``set_times_visible`` shows or hides the gutter.
    """

    GAP = 14  # between the times and the text, in pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gutter = _TimeGutter(self)
        self._times = True
        self._time_color = QColor("#71717a")
        self.updateRequest.connect(self._scroll_gutter)
        self._fit_gutter()

    # --- what the window and the tests ask
    def times_visible(self):
        return self._times

    def gutter_width(self):
        """The room the gutter takes: the width of "00:00:00" in this font and a gap, or 0 while it is hidden."""
        return self.fontMetrics().horizontalAdvance("00:00:00") + self.GAP if self._times else 0

    def time_of_block(self, number):
        """The time drawn for block ``number`` ("" for none)."""
        return clock_text(self.document().findBlockByNumber(number).userState())

    def set_times_visible(self, on):
        self._times = bool(on)
        self._fit_gutter()

    def set_time_color(self, color):
        self._time_color = QColor(color)
        self._gutter.update()

    # --- the gutter's place and size
    def _fit_gutter(self):
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)
        self._place_gutter()

    def _place_gutter(self):
        view = self.viewport().geometry()
        width = self.gutter_width()
        self._gutter.setGeometry(QRect(view.left() - width, view.top(), width, view.height()))
        self._gutter.setVisible(width > 0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_gutter()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange and hasattr(self, "_gutter"):
            self._fit_gutter()  # another font is another width

    def _scroll_gutter(self, rect, dy):
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())

    def paint_times(self, gutter, event):
        painter = QPainter(gutter)
        painter.setFont(self.font())
        painter.setPen(self._time_color)
        height = self.fontMetrics().height()
        block = self.firstVisibleBlock()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        while block.isValid() and top <= event.rect().bottom():
            bottom = top + round(self.blockBoundingRect(block).height())
            if block.isVisible() and bottom >= event.rect().top():
                text = clock_text(block.userState())
                if text:
                    painter.drawText(QRect(0, top, gutter.width() - self.GAP, height), Qt.AlignmentFlag.AlignLeft, text)
            block = block.next()
            top = bottom
        painter.end()


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
        self._sgr = {}               # the colour each stream's last line left on, which its next line starts in
        self._run_formats = {}       # (style, kind) -> the format of a run of coloured text
        self._empty = True           # nothing is drawn yet
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
        self.times_switch = ToggleSwitch(QCoreApplication.translate("Console", "Times"), QCoreApplication.translate("Console", "Times"))
        self.times_switch.setToolTip(QCoreApplication.translate("Console", "Show the time of each line, beside the text"))
        self.copy_button =self._button("copy", QCoreApplication.translate("Console", "Copy the selection, or all the output shown"))
        self.save_button = self._button("download", QCoreApplication.translate("Console", "Save the output shown to a file…"))
        self.clear_button = self._button("trash-2", QCoreApplication.translate("Console", "Clear the view (the log file keeps everything)"))
        self.log_button = self._button("file-text", QCoreApplication.translate("Console", "Open the log file"))
        self.folder_button = self._button("folder-open", QCoreApplication.translate("Console", "Show the log folder"))
        for widget in (self.previous_button, self.next_button, self.copy_button, self.save_button,
                       self.clear_button, self.log_button, self.folder_button):
            widget.setFixedWidth(36)
        bar.addWidget(self.search, 1)
        for widget in (self.previous_button, self.next_button, self.match_label, self.filter_box,
                       self.follow_switch, self.times_switch, self.copy_button, self.save_button, self.clear_button,
                       self.log_button, self.folder_button):
            bar.addWidget(widget)
        root.addLayout(bar)

        self.text = ConsoleText()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setMaximumBlockCount(int(capture.lines.maxlen or Console_Capture.RING_LINES))
        self.text.setFont(qt_monospace_font(self.text.font()))
        self.times_switch.setChecked(saved_show_times())
        self.text.set_times_visible(self.times_switch.isChecked())
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
        self.times_switch.toggled.connect(self._set_times)
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
        self._surface = tokens["surface"]
        self._base_colors = {INFO: tokens["text_soft"], WARNING: tokens["warning"], ERROR: tokens["danger"]}
        self.formats = {kind: self._format(color) for kind, color in self._base_colors.items()}
        self.text.set_time_color(tokens["text_subtle"])
        self._run_formats = {}
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
        self._empty = True
        self._sgr = {}
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
            runs, visible = self._read_colours(line)
            kind = self._classifier.kind(visible, remember=not line.partial)
            if not self._wanted(kind):
                continue
            if line.seq == self._last_seq:  # the next state of a progress line replaces it
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
            elif not self._empty:
                cursor.insertBlock()
            coloured = any(style.has_color for _, style in runs)
            for run, style in runs:
                cursor.insertText(run, self._run_format(style, INFO if coloured else kind))
            cursor.block().setUserState(seconds_of_day(line.time))  # drawn in the gutter, not part of the text
            self._empty = False
            self._last_seq = line.seq
        cursor.endEditBlock()
        self._setting_scroll = True
        scroll.setValue(scroll.maximum() if self._following else kept)
        self._setting_scroll = False
        if self.search.text():
            self._highlight_matches()
        self._update_footer()

    def _read_colours(self, line):
        """(runs, visible): the line as [(text, Style), ...] with its colour sequences read, and the text that shows.

        A line starts in the colour the stream's last line left on, as in a terminal, so a message of several
        lines inside one colour is in it throughout. A progress state (partial) does not change that colour.
        """
        before = self._sgr.get(line.stream, DEFAULT_STYLE)
        if ESC not in line.text and before == DEFAULT_STYLE:
            return ([(line.text, DEFAULT_STYLE)] if line.text else []), line.text
        runs, after = parse_ansi(line.text, before)
        if not line.partial:
            self._sgr[line.stream] = after
        return runs, "".join(text for text, _ in runs)

    def _run_format(self, style, base):
        """The format of a run: ``base`` (the line's INFO, WARNING or ERROR) where the style gives no colour."""
        if style == DEFAULT_STYLE:
            return self.formats[base]
        key = (style, base)
        found = self._run_formats.get(key)
        if found is None:
            found = self._run_formats[key] = self._style_format(style, base)
        return found

    def _style_format(self, style, base):
        """A colour sequence's style as a text format that shows on this page, in light and in dark."""
        text = self._base_colors[base]
        page = {"dark": self.dark, "background": self._surface}
        if style.reverse:
            # The text takes the background's colour (the page's, when there is none) and the background the text's.
            foreground = palette_color(style.bg, dark=None) if style.bg else self._surface
            background = resolve_color(style.fg, **page) if style.fg else text
        elif style.bg:
            # A background of its own comes with a text colour chosen to go with it: both in the terminal's shades.
            foreground = palette_color(style.fg, dark=None) if style.fg else text
            background = palette_color(style.bg, dark=None)
        else:
            foreground = resolve_color(style.fg, **page) if style.fg else text
            background = None
        if background:
            foreground = readable_color(foreground, background)  # and the text must show on that colour
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(foreground))
        if background:
            fmt.setBackground(QColor(background))
        if style.bold:
            fmt.setFontWeight(QFont.Weight.Bold)
        if style.italic:
            fmt.setFontItalic(True)
        if style.underline:
            fmt.setFontUnderline(True)
        return fmt

    def _update_footer(self):
        shown = 0 if self._empty else self.text.blockCount()
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

    def _set_times(self, on):
        """The Times switch: show or hide the gutter, and keep the choice for the next Console."""
        self.text.set_times_visible(on)
        save_show_times(on)

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
        """The text of every line shown, without the times in the gutter."""
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

