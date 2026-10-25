# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The start-up splash: a window that is up a third of a second after a click, and follows the start.

Layout: the kicker, "EMAP-SSN" and the window's name at the left, the copyright and version
under them, the network art at the right, and below the text a status line with a progress
bar. The colours are the Studio theme's light shades (desktop/Studio_Theme.py) in the default
purple, whatever theme colour the windows use: the splash is the program's, not a window's.

The status line follows the start. The launcher (desktop/Desktop_Launcher.py) and the window
it starts report what they are doing as JSON lines in a status file (utilities/Startup_Status.py);
the splash reads the file as it grows, translates each text in the saved language (the catalogs'
"Splash" context) and shows it. A window that starts fills the bar as it passes its milestones;
a bar with no milestone yet sweeps. The splash closes when the window writes the ready file.

Show details expands the splash downward into a section that shows what the start prints, live: the
dependency check's lines, then what the window itself prints (the monitor keeps it in application.log,
which the splash follows), in the splash's own light style, warnings and errors in its colours, the
newest line in view. Copy and Open log are there. A splash that is collapsed when the window is ready
closes, as it always has. One that is expanded stays: the user is watching, so it says Ready and offers
Close, and closes when that is pressed.

If the window ends before it is ready, the splash opens that section by itself and says why, with
Close: the error view. A step of the start that goes wrong without stopping it (show_warning) opens the
section by itself too, with that step's lines in the warning colour; the start goes on, and the splash,
open, stays at Ready until Close is pressed, as it does for anyone who opened the section.

A program that sits beside this one adds its own window with register_kind (its name under the
product's name, its art), and its own texts with install_extra_catalogs; this file names none.

It imports only PySide6's QtCore, QtGui and QtWidgets and the standard library, so it is on
screen long before the heavy imports (NumPy, torch, VisPy) of the windows it waits for, and it
runs in a process of its own: the windows load torch before Qt on purpose. It has no taskbar
button (a Qt splash-screen window is a tool window on Windows) and opens on the screen under
the mouse pointer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path

T_START = time.time()  # the splash's own clock, for the first-paint measure

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtCore import QCoreApplication, QLocale, QPointF, QRectF, QSize, Qt, QTimer, QTranslator, QUrl, Signal
from PySide6.QtGui import (
    QColor, QCursor, QDesktopServices, QFont, QFontDatabase, QFontMetricsF, QGuiApplication, QIcon, QImage, QPainter,
    QPainterPath, QPen, QPixmap, QRadialGradient, QTextCharFormat, QTextCursor,
)
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QPushButton, QWidget

from utilities import Startup_Status as Status
from utilities.Console_Capture import Splitter
from utilities.Line_Kinds import ERROR, WARNING, Classifier

LOGOS = SRC_DIR / "bin" / "logos"
FONTS = SRC_DIR / "resources" / "fonts" / "desktop"
LANGUAGES = SRC_DIR / "resources" / "languages"
APP_SETTINGS_VARIABLE = "SSN_APP_SETTINGS_PATH"

PRODUCT = "EMAP-SSN"
VERSION = "0.4.0"
# What each kind of window is called under the product's name, and which art it takes.
SUBTITLES = {
    "config": Status.QT_TRANSLATE_NOOP("Splash", "Configuration"),
    "tools": Status.QT_TRANSLATE_NOOP("Splash", "Tools"),
    "viewer": Status.QT_TRANSLATE_NOOP("Splash", "Viewer"),
}
KINDS = tuple(SUBTITLES)
LOGO_FILES = {"config": "splash_logo_viewer_512.png", "tools": "splash_logo_tool_512.png",
              "viewer": "splash_logo_viewer_512.png"}


def register_kind(kind, subtitle, logo):
    """Add a window of a program beside this one: ``subtitle`` is its name under the product's, in English and
    marked for translation by the program (the "Splash" context of its catalog), and ``logo`` is the path of its
    art, a 512 px copy of its logo. A second call with the same ``kind`` replaces the first."""
    SUBTITLES[kind] = subtitle
    LOGO_FILES[kind] = str(logo)

# Studio_Theme's NEUTRAL_SHADES["light"] and its default purple accent. tests/test_splash.py
# compares them with the theme's own, so the splash cannot drift from it.
SCHEME = {
    "surface": "#ffffff", "panel": "#f4f4f5", "border": "#e4e4e7", "text": "#09090b", "text_soft": "#3f3f46",
    "muted": "#52525b", "subtle": "#71717a", "track": "#e4e4e7", "accent": "#5b5bd6", "field_border": "#dcdce0",
    "danger": "#dc2626", "warning": "#b45309", "hover": "#f4f4f5", "accent_soft": "#e9ecfc",
}

WIDTH, HEIGHT = 640, 360
# The section Show details opens, below the splash: the output's view and a row of buttons.
DETAILS_HEIGHT = 264
DETAILS_MARGIN = 32
DETAILS_GAP = 12
BUTTON_HEIGHT = 34
# How many lines the section keeps; the start prints a few dozen, a failing one a traceback.
MAX_DETAIL_LINES = 5000
# How many columns of the section's monospace font fit in it, where a warning wraps.
WARNING_COLUMNS = 78
# How much of the log one tick reads, so a burst of output never stalls the splash.
TAIL_READ_LIMIT = 256 * 1024
RADIUS = 14.0
# The bar: the text column's width, 3 px high.
TEXT_X, TEXT_WIDTH = 40.0, 300.0
BAR_Y, BAR_HEIGHT = 318.0, 3.0
# The window may be this many seconds without news before the bar creeps on its own, and how far it may creep.
CREEP_SECONDS, CREEP_LIMIT = 1.0, 0.12
TICK_MS = 33
# The milestones the launcher itself reports (the windows report theirs with Startup_Status).
FRACTION_CHECKING, FRACTION_CHECKED, FRACTION_STARTED = 0.06, 0.34, 0.40


# ---------------------------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------------------------

def app_settings_path(environment=None):
    environment = os.environ if environment is None else environment
    named = environment.get(APP_SETTINGS_VARIABLE, "").strip()
    return Path(named) if named else SRC_DIR.parent / "app_settings.json"


def saved_language_setting(environment=None):
    """The LANGUAGE setting of app_settings.json: "system", "en" or a code ("system" when unset or unreadable)."""
    try:
        value = json.loads(app_settings_path(environment).read_text(encoding="utf-8")).get("LANGUAGE")
    except (OSError, ValueError, AttributeError):
        return "system"
    return value if isinstance(value, str) and value else "system"


def catalog_path(code, languages_dir=LANGUAGES):
    return Path(languages_dir) / code / f"emapssn_{code}.qm"


def splash_language(setting=None, ui_languages=None, languages_dir=LANGUAGES):
    """The code of the language the windows show, or None for English: what desktop.Desktop_App.resolve_language gives.

    "system" follows the system's display languages where a catalog exists; a language whose
    compiled catalog is gone shows English.
    """
    setting = saved_language_setting() if setting is None else setting
    if setting == "en":
        return None
    if setting == "system":
        tags = QLocale.system().uiLanguages() if ui_languages is None else ui_languages
        for tag in tags:
            name = QLocale(tag).name()
            for code in (name, name.split("_")[0]):
                if code == "en":
                    return None
                if catalog_path(code, languages_dir).is_file():
                    return code
        return None
    return setting if catalog_path(setting, languages_dir).is_file() else None


def language_font(code, languages_dir=LANGUAGES):
    """(family, files) of the font a language brings besides the core faces (its language.json), or (None, [])."""
    try:
        font = json.loads((Path(languages_dir) / code / "language.json").read_text(encoding="utf-8"))["font"]
        return font["family"], [FONTS / name for name in font["files"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None, []


def language_font_files(code, languages_dir=LANGUAGES):
    """The bundled font files a language needs besides the core faces, as paths."""
    return language_font(code, languages_dir)[1]


def load_fonts(language=None):
    """Register the bundled Noto faces the splash draws in (and the language's, if it brings its own)."""
    files = [FONTS / "noto" / "NotoSans" / "NotoSans-Regular.ttf", FONTS / "noto" / "NotoSans" / "NotoSans-SemiBold.ttf",
             FONTS / "noto" / "NotoSansMono" / "NotoSansMono-Regular.ttf"]
    _language_families.clear()
    if language:
        family, extra = language_font(language)
        files += extra
        if family:
            _language_families.append(family)
    for path in files:
        QFontDatabase.addApplicationFont(str(path))


def install_catalog(language):
    """Install the language's compiled catalog; returns the translator (keep it alive), or None."""
    if not language:
        return None
    translator = QTranslator()
    if translator.load(str(catalog_path(language))):
        QCoreApplication.installTranslator(translator)
        return translator
    return None


def install_extra_catalogs(language, catalogs):
    """Install the catalogs a program beside this one adds, as (directory, name) pairs: the language's
    <directory>/<language>/<name>_<language>.qm. Returns the translators (keep them alive); one that is
    missing leaves its texts English."""
    translators = []
    if not language:
        return translators
    for directory, name in catalogs:
        translator = QTranslator()
        if translator.load(str(Path(directory) / language / f"{name}_{language}.qm")):
            QCoreApplication.installTranslator(translator)
            translators.append(translator)
    return translators


def tr(text):
    return QCoreApplication.translate("Splash", text)


# ---------------------------------------------------------------------------------------------
# The status file
# ---------------------------------------------------------------------------------------------

class StatusFeed:
    """The JSON lines of a status file, as they are appended: poll() gives the new, complete ones."""

    def __init__(self, path):
        self.path = None if path is None else Path(path)
        self._offset = 0
        self._partial = b""

    def poll(self):
        if self.path is None:
            return []
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self._offset)
                data = handle.read()
        except OSError:
            return []
        self._offset += len(data)
        data = self._partial + data
        *lines, self._partial = data.split(b"\n")
        messages = []
        for line in lines:
            try:
                message = json.loads(line.decode("utf-8", "replace"))
            except ValueError:
                continue
            if isinstance(message, dict) and isinstance(message.get("text"), str):
                messages.append(message)
        return messages


def tail_lines(path, count=14, limit=64 * 1024):
    """The last ``count`` non-empty lines of a text file, read from its last ``limit`` bytes ([] if unreadable)."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            data = handle.read()
    except OSError:
        return []
    lines = [line.rstrip("\r") for line in data.decode("utf-8", "replace").split("\n")]
    return [line for line in lines if line.strip()][-count:]


class LogTail:
    """A log file another process appends to, as the lines it gains: poll() gives the new ones.

    The window's output reaches the splash this way: the monitor (desktop/Desktop_Launcher_Monitor.py)
    runs the window with its output in application.log, which the window's capture
    (utilities/Console_Capture.py) writes as it prints, so the file is the one channel that is there
    for every window and every kind of output, and the splash needs nothing of the window but the folder
    it was given. Lines come as (text, partial) pairs, as utilities.Console_Capture.Splitter gives them: a
    progress bar that rewrites its line is one partial line that changes, finished by its newline.
    """

    def __init__(self, path=None):
        self.path = None
        self._offset = 0
        self._splitter = Splitter()
        if path is not None:
            self.follow(path)

    def follow(self, path):
        self.path = Path(path)
        self._offset = 0
        self._splitter = Splitter()

    def poll(self):
        if self.path is None:
            return []
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self._offset)
                data = handle.read(TAIL_READ_LIMIT)
        except OSError:
            return []
        if not data:
            return []
        self._offset += len(data)
        return self._splitter.feed(data)

    def finish(self):
        """Everything the file has now, including a last line with no newline."""
        lines = []
        while True:
            more = self.poll()
            if not more:
                break
            lines.extend(more)
        lines.extend(self._splitter.finish())
        return lines


# ---------------------------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------------------------

# The families a language brings (its language.json), tried after the core one for the glyphs it lacks.
_language_families = []


def _font(pixels, bold=False, family="Noto Sans"):
    font = QFont(family)
    if _language_families:
        font.setFamilies([family, *_language_families])
    font.setPixelSize(int(pixels))
    font.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
    return font


def _fitted_font(text, width, pixels, smallest, bold=True):
    """The bold font of ``pixels``, or the largest size down to ``smallest`` that fits ``text`` in ``width``."""
    font = _font(pixels, bold)
    while pixels > smallest and QFontMetricsF(font).horizontalAdvance(text) > width:
        pixels -= 1
        font = _font(pixels, bold)
    return font


def screen_center(point=None):
    """The centre of the available area of the screen under ``point`` (the mouse pointer by default)."""
    screen = QGuiApplication.screenAt(QCursor.pos() if point is None else point) or QGuiApplication.primaryScreen()
    return screen.availableGeometry().center()


def _chevron(up, color, size=12):
    """A small chevron, pointing up or down, as an icon in ``color``."""
    ratio = 2
    pixmap = QPixmap(size * ratio, size * ratio)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor(color), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    near, far = (8.0, 4.0) if up else (4.0, 8.0)
    painter.drawPolyline([QPointF(2.5, near), QPointF(6.0, far), QPointF(9.5, near)])
    painter.end()
    return QIcon(pixmap)


class DetailsView(QPlainTextEdit):
    """The section's text: the start's output, a line to a block, each in the colour of what it says.

    Plain lines are in the splash's soft text colour, warnings and errors in its warning and danger
    colours (utilities/Line_Kinds.py decides which a line is, and an indented line under a problem has
    its colour). The newest line stays in view while the view is at its end; scrolled up, it stays
    where the reader put it, and following resumes at the end again. A partial line, a progress bar's
    latest state, is replaced by the next until a newline finishes it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setMaximumBlockCount(MAX_DETAIL_LINES)
        font = _font(11, family="Noto Sans Mono")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setStyleSheet(
            f"QPlainTextEdit {{ background: {SCHEME['panel']}; color: {SCHEME['text_soft']};"
            f" border: 1px solid {SCHEME['border']}; border-radius: 8px; padding: 6px;"
            f" selection-background-color: {SCHEME['accent_soft']}; selection-color: {SCHEME['text']}; }}"
        )
        self.formats = {}
        for kind, name in (("info", "text_soft"), (WARNING, "warning"), (ERROR, "danger")):
            line_format = QTextCharFormat()
            line_format.setForeground(QColor(SCHEME[name]))
            self.formats[kind] = line_format
        self._classifier = Classifier()
        self._empty = True
        self._partial = False

    def is_empty(self):
        return self._empty

    def kind_of(self, line_number):
        """The kind ("info", "warning" or "error") a line was shown as, by the colour it has (tests)."""
        block = self.document().findBlockByNumber(line_number)
        colour = block.begin().fragment().charFormat().foreground().color().name()
        for kind, line_format in self.formats.items():
            if line_format.foreground().color().name() == colour:
                return kind
        return None

    def add_lines(self, items, kind=None):
        """Append (text, partial) pairs, keeping the newest line in view if the view was at its end.

        ``kind`` ("info", "warning" or "error") puts every line in that kind's colour, whatever it says: a
        step that went wrong as a whole shows its lines as a warning, though they read as plain output.
        """
        items = list(items)
        if not items:
            return
        bar = self.verticalScrollBar()
        following = bar.value() >= bar.maximum() - 3
        cursor = QTextCursor(self.document())
        cursor.beginEditBlock()
        for text, partial in items:
            cursor.movePosition(QTextCursor.MoveOperation.End)
            if self._partial:
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
            elif not self._empty:
                cursor.insertBlock()
            found = self._classifier.kind(text, remember=not partial)
            cursor.insertText(text, self.formats[kind or found])
            self._empty = False
            self._partial = partial
        cursor.endEditBlock()
        if following:
            bar.setValue(bar.maximum())


class SplashWindow(QWidget):
    """The splash. ``kind`` is "config", "tools" or "viewer"; ``language`` a code or None."""

    closed = Signal()

    def __init__(self, kind="config", language=None, status_file=None):
        super().__init__(None, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.NoDropShadowWindowHint)
        if kind not in SUBTITLES:
            raise ValueError(f"unknown splash kind {kind!r}")
        self.kind = kind
        self.language = language
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)  # the rounded corners
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFixedSize(WIDTH, HEIGHT)
        self.colors = {name: QColor(value) for name, value in SCHEME.items()}
        self.logo = QPixmap(str(LOGOS / LOGO_FILES[kind]))  # a 512 px copy: the shipped art decodes to 75 MB
        self.status = tr(Status.STARTING)
        self.target_fraction = None   # None: the sweeping bar; a number: the share done at the last news
        self.shown_fraction = 0.0
        self._news_time = time.monotonic()
        self._started = self._news_time
        self.error = False
        self.error_title = ""
        self.ready = False            # the window is up and the section is open: the user closes the splash
        self.details_open = False     # the section Show details opens
        self._shifted = 0             # how far the window moved up to keep the open section on the screen
        self.log_path = None
        self.feed = StatusFeed(status_file)
        self.tail = LogTail()
        self.log_view = self.details = DetailsView(self)
        self.details_button = self._make_link(self._details_text(), self.toggle_details)
        self.copy_button = self._make_button(QCoreApplication.translate("Splash", "Copy"), self.copy_error)
        self.log_button = self._make_button(QCoreApplication.translate("Splash", "Open log"), self.open_log)
        self.close_button = self._make_button(QCoreApplication.translate("Splash", "Close"), self.close)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(TICK_MS)
        self.move_to(screen_center())
        self._apply_layout()

    # --- placing
    def move_to(self, center):
        self.move(center.x() - self.width() // 2, center.y() - self.height() // 2)

    # --- status
    def show_text(self, text, fraction=None):
        """Show ``text`` as it is (a line the dependency check printed); fraction None keeps the bar's milestone."""
        self.status = text
        if fraction is not None:
            self.set_fraction(fraction)
        self.update()

    def set_status(self, text, fraction=None, **values):
        """Show a reported status: ``text`` is English; it is translated, then filled in with ``values``."""
        shown = tr(text)
        try:
            shown = shown.format(**values) if values else shown
        except (KeyError, IndexError, ValueError):
            pass
        self.show_text(shown, fraction)

    def set_fraction(self, fraction):
        self.target_fraction = max(self.target_fraction or 0.0, min(1.0, float(fraction)))  # never backwards
        self._news_time = time.monotonic()

    def apply(self, messages):
        for message in messages:
            fraction = message.get("fraction")
            self.set_status(message["text"], fraction if isinstance(fraction, (int, float)) else None,
                            **(message.get("values") if isinstance(message.get("values"), dict) else {}))

    def _tick(self):
        if not self.error and not self.ready:
            self.apply(self.feed.poll())
            if self.target_fraction is not None:
                idle = time.monotonic() - self._news_time
                creep = min(CREEP_LIMIT, max(0.0, idle - CREEP_SECONDS) * 0.02)
                goal = min(0.97, self.target_fraction + creep)
                self.shown_fraction += (goal - self.shown_fraction) * 0.18
        if not self.error:  # an error has had its last read (sync_log)
            self.details.add_lines(self.tail.poll())
        self.update()

    # --- what the start prints
    def add_output(self, text, partial=False, kind=None):
        """Add one line of the start's output to the section (the dependency check's lines come this way)."""
        self.details.add_lines([(text, partial)], kind=kind)

    def show_warning(self, lines):
        """A step went wrong and the start goes on: its lines, in the warning colour, and the section open.

        The section opens by itself so the lines are read, as an error view's does; unlike one, it can be
        closed again, and the start goes on to Ready. A warning's lines are sentences, so they wrap to the
        section's width (a log's lines do not: a traceback keeps its columns).
        """
        shown = []
        for line in lines:
            indent = " " * (len(line) - len(line.lstrip()))
            shown.extend(textwrap.wrap(line, WARNING_COLUMNS, subsequent_indent=indent + "  ", break_long_words=False,
                                       break_on_hyphens=False) or [""])
        self.details.add_lines([(line, False) for line in shown], kind=WARNING)
        self.set_details_open(True)

    def follow_log(self, path):
        """Follow the window's output: the file it is kept in (application.log), as it grows."""
        self.tail.follow(path)
        self.log_path = str(path)
        self._apply_layout()

    def sync_log(self):
        """Read what the log holds now, down to a last line with no newline."""
        self.details.add_lines(self.tail.finish())

    # --- the section
    @staticmethod
    def _details_text(opened=False):
        return (QCoreApplication.translate("Splash", "Hide details") if opened
                else QCoreApplication.translate("Splash", "Show details"))

    def toggle_details(self):
        self.set_details_open(not self.details_open)

    def set_details_open(self, opened):
        """Open or close the section. It stays open once the splash is an error view or says Ready."""
        if (self.error or self.ready) and not opened:
            return
        self.details_open = bool(opened)
        self._apply_layout()
        if opened:
            self.details.verticalScrollBar().setValue(self.details.verticalScrollBar().maximum())

    def _section_height(self):
        """The window's height with the section open: the whole of it, unless the screen has less room."""
        screen = QGuiApplication.screenAt(self.geometry().center()) or QGuiApplication.primaryScreen()
        room = screen.availableGeometry().height() - 16
        return max(HEIGHT + 120, min(HEIGHT + DETAILS_HEIGHT, room))

    def _apply_layout(self):
        """Size the window and place the widgets for what it shows: collapsed, the section open, Ready or an error."""
        opened = self.details_open
        height = self._section_height() if opened else HEIGHT
        self.setFixedSize(WIDTH, height)
        self._keep_on_screen(opened)
        locked = self.error or self.ready
        link = self.details_button
        link.setText(self._details_text(opened))
        link.setIcon(_chevron(opened, SCHEME["accent"]))
        link.adjustSize()
        link.setGeometry(int(TEXT_X) - 8, 328, link.width(), 26)
        link.setVisible(not locked)
        buttons_y = height - 16 - BUTTON_HEIGHT
        top = HEIGHT + DETAILS_GAP
        self.details.setGeometry(DETAILS_MARGIN, top, WIDTH - 2 * DETAILS_MARGIN, buttons_y - 10 - top)
        right = WIDTH - DETAILS_MARGIN
        for button in (self.close_button, self.log_button, self.copy_button):
            width = max(button.sizeHint().width(), 84)
            button.setGeometry(right - width, buttons_y, width, BUTTON_HEIGHT)
            right -= width + 10
        self.log_button.setEnabled(self.log_path is not None)
        self.details.setVisible(opened)
        self.copy_button.setVisible(opened)
        self.log_button.setVisible(opened)
        self.close_button.setVisible(opened and locked)
        self.update()

    def _keep_on_screen(self, opened):
        """Grow downward; if that would leave the screen, move up by as much (and back down when it closes)."""
        if opened:
            screen = QGuiApplication.screenAt(self.geometry().center()) or QGuiApplication.primaryScreen()
            overflow = self.y() + self.height() - screen.availableGeometry().bottom()
            if overflow > 0:
                self.move(self.x(), self.y() - overflow)
                self._shifted += overflow
        elif self._shifted:
            self.move(self.x(), self.y() + self._shifted)
            self._shifted = 0

    # --- the end of the start
    def show_ready(self):
        """The window is up and the section is open: the user is watching, so say so and wait for Close."""
        self.ready = True
        self.status = QCoreApplication.translate("Splash", "Ready")
        self.target_fraction = 1.0
        self.shown_fraction = 1.0
        self.sync_log()
        self._apply_layout()
        self.close_button.setFocus()
        self.update()

    def show_error(self, lines=None, log_path=None, reason=None):
        """Turn into the error view: the section opens by itself, says why it stopped, and offers Copy, Open log and Close.

        ``lines``, if given, are put in the section when it has nothing yet (the start failed before
        the log was followed).
        """
        self.error = True
        self.details_open = True
        if log_path is not None:
            self.log_path = str(log_path)
        self.error_title = reason or QCoreApplication.translate("Splash", "{window} stopped before its window opened.").format(
            window=f"{PRODUCT} {tr(SUBTITLES[self.kind])}")
        if lines and self.details.is_empty():
            self.details.add_lines((line, False) for line in lines)
        if self.details.is_empty():
            self.details.add_lines([(QCoreApplication.translate("Splash", "(It printed nothing.)"), False)])
        self._apply_layout()
        self.details.verticalScrollBar().setValue(self.details.verticalScrollBar().maximum())
        self.close_button.setFocus()
        self.activateWindow()
        self.raise_()
        self.update()

    def _make_button(self, text, slot):
        button = QPushButton(text, self)
        button.setFont(_font(12))
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setStyleSheet(
            f"QPushButton {{ background: {SCHEME['surface']}; color: {SCHEME['text']}; border: 1px solid {SCHEME['field_border']};"
            f" border-radius: 8px; padding: 6px 16px; }}"
            f"QPushButton:hover {{ background: {SCHEME['hover']}; }}"
            f"QPushButton:focus {{ border: 2px solid {SCHEME['accent']}; padding: 5px 15px; }}"
        )
        button.clicked.connect(slot)
        return button

    def _make_link(self, text, slot):
        """The Show details / Hide details button: text in the accent, no frame, as the Viewer's links are."""
        button = QPushButton(text, self)
        button.setFont(_font(12, True))
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setIconSize(QSize(12, 12))
        button.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {SCHEME['accent']}; border: none; border-radius: 6px;"
            f" padding: 3px 8px 3px 6px; text-align: left; }}"
            f"QPushButton:hover {{ background: {SCHEME['accent_soft']}; }}"
            f"QPushButton:focus {{ background: {SCHEME['accent_soft']}; }}"
        )
        button.clicked.connect(slot)
        return button

    def error_text(self):
        return self.error_title + "\n\n" + self.details.toPlainText()

    def copy_error(self):
        """Copy the section's text (with the reason first, in an error view); Copied shows for a moment."""
        QGuiApplication.clipboard().setText(self.error_text() if self.error else self.details.toPlainText())
        self.copy_button.setText(QCoreApplication.translate("Splash", "Copied"))
        QTimer.singleShot(1500, lambda: self.copy_button.setText(QCoreApplication.translate("Splash", "Copy")))

    def open_log(self):
        if not self.log_path:
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(self.log_path)):  # nothing opens .log files: show the folder
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.log_path).parent)))

    def keyPressEvent(self, event):
        if (self.error or self.ready) and event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
        self.closed.emit()

    # --- painting
    def paintEvent(self, event):
        c = self.colors
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        height = self.height()
        outer = QRectF(0.5, 0.5, self.width() - 1, height - 1)
        shape = QPainterPath()
        shape.addRoundedRect(outer, RADIUS, RADIUS)
        painter.setClipPath(shape)
        painter.fillRect(self.rect(), c["surface"])
        self._paint_start(painter)
        painter.setClipping(False)
        painter.setPen(QPen(c["border"], 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(outer, RADIUS, RADIUS)
        painter.end()

    def _paint_start(self, painter):
        c = self.colors
        # A soft light of the accent behind the art, then the art.
        glow = QRadialGradient(QPointF(WIDTH - 180, HEIGHT / 2), 230)
        halo = QColor(c["accent"])
        halo.setAlphaF(0.14)
        glow.setColorAt(0.0, halo)
        halo.setAlphaF(0.0)
        glow.setColorAt(1.0, halo)
        painter.fillRect(QRectF(0, 0, WIDTH, HEIGHT), glow)
        if not self.logo.isNull():
            painter.drawPixmap(QRectF(WIDTH - 330, 36, 300, 300), self.logo, QRectF(self.logo.rect()))
        x, width = TEXT_X, TEXT_WIDTH
        left, vcenter = Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignVCenter
        painter.setPen(c["accent"])
        painter.setFont(_font(15, True))
        painter.drawText(QRectF(x, 36, width + 10, 22), left | vcenter, QCoreApplication.translate("Splash", "Protein Sequence Similarity Networks"))
        painter.setPen(c["text"])
        painter.setFont(_font(44, True))
        painter.drawText(QRectF(x, 58, width, 56), left | vcenter, PRODUCT)
        subtitle = tr(SUBTITLES[self.kind])
        painter.setFont(_fitted_font(subtitle, width, 30, 20))
        painter.drawText(QRectF(x, 112, width, 40), left | vcenter, subtitle)
        painter.setPen(c["muted"])
        painter.setFont(_font(11))
        painter.drawText(
            QRectF(x, 204, width, 44), left | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            QCoreApplication.translate("Splash", "© 2026 Xuebin Feng, University of Toronto. Apache License 2.0.") + "  "
            + QCoreApplication.translate("Splash", "Version {version}").format(version=VERSION),
        )
        if self.error:
            # Why it stopped, in the place of the status line and the bar.
            painter.setPen(c["danger"])
            painter.setFont(_font(13, True))
            painter.drawText(QRectF(x, 262, width + 10, 54), left | vcenter | Qt.TextFlag.TextWordWrap, self.error_title)
            return
        painter.setPen(c["text"])
        font = _font(12, self.ready)
        painter.setFont(font)
        status = QFontMetricsF(font).elidedText(self.status, Qt.TextElideMode.ElideMiddle, width + 10)
        painter.drawText(QRectF(x, 286, width + 10, 22), left | vcenter, status)
        self._paint_bar(painter)

    def _paint_bar(self, painter):
        c = self.colors
        bar = QRectF(TEXT_X, BAR_Y, TEXT_WIDTH, BAR_HEIGHT)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(c["track"])
        painter.drawRoundedRect(bar, 1.5, 1.5)
        painter.setBrush(c["accent"])
        if self.target_fraction is None:
            period = 1.4
            phase = ((time.monotonic() - self._started) % period) / period
            span = bar.width() * 0.28
            left = bar.left() + (bar.width() + span) * phase - span
            start, end = max(bar.left(), left), min(left + span, bar.right())
            if end > start:
                painter.drawRoundedRect(QRectF(start, bar.top(), end - start, bar.height()), 1.5, 1.5)
        else:
            painter.drawRoundedRect(
                QRectF(bar.left(), bar.top(), max(3.0, bar.width() * min(1.0, self.shown_fraction)), bar.height()),
                1.5, 1.5)


# ---------------------------------------------------------------------------------------------
# Standalone use: python src/desktop/Splash.py (screenshots, and the hand-off tests)
# ---------------------------------------------------------------------------------------------

def render(window, scheme_background=None):
    """The window as it looks on the page behind it, as an image (a translucent window shows nothing alone)."""
    pixmap = window.grab()
    image = QImage(pixmap.size(), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(scheme_background or SCHEME["panel"]))
    image.setDevicePixelRatio(pixmap.devicePixelRatio())
    painter = QPainter(image)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kind", default="config", choices=KINDS)
    parser.add_argument("--language", default=None, help="a language code; the saved setting by default; en for English")
    parser.add_argument("--status", default=None, help="a status text (English; it is translated), as a window reports it")
    parser.add_argument("--value", action="append", default=[], metavar="NAME=TEXT", help="a value for the status's {NAME}")
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--status-file", default=None)
    parser.add_argument("--ready-file", default=None)
    parser.add_argument("--hidden", action="store_true", help="draw nowhere (tests)")
    parser.add_argument("--shot", default=None, help="render once to this PNG and exit")
    parser.add_argument("--error-log", default=None, help="show the error view with the end of this file")
    parser.add_argument("--details", action="store_true", help="start with the section open")
    parser.add_argument("--tail", default=None, metavar="FILE", help="follow this log file in the section")
    parser.add_argument("--ready", action="store_true", help="show the Ready state (with --details)")
    args = parser.parse_args(argv)
    application = QApplication(sys.argv[:1])
    language = splash_language(args.language if args.language else None)
    if args.language == "en":
        language = None
    load_fonts(language)
    translator = install_catalog(language)  # noqa: F841 (kept alive)
    window = SplashWindow(args.kind, language, args.status_file)
    if args.status:
        window.set_status(args.status, args.fraction, **dict(item.split("=", 1) for item in args.value if "=" in item))
        if args.fraction is not None:
            window.shown_fraction = args.fraction  # a shot has no time to ease in
    if args.tail:
        window.follow_log(args.tail)
    if args.details:
        window.set_details_open(True)
    if args.error_log:
        window.show_error(tail_lines(args.error_log), args.error_log)
    if args.hidden or args.shot:
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.show()
    application.processEvents()
    print(json.dumps({"first_paint_ms": round((time.time() - T_START) * 1000, 1), "language": language}), flush=True)
    if args.shot:
        window._tick()  # one read of the log, as the timer would
        if args.ready and args.details:
            window.show_ready()
        render(window).save(args.shot)
        return 0
    if args.ready_file:
        ready = Path(args.ready_file)
        poll = QTimer()

        def when_ready():
            # Collapsed, the splash is done; with the section open the user is watching: say Ready and wait for Close.
            if ready.exists() and not window.error and not window.ready:
                window.show_ready() if window.details_open else window.close()

        poll.timeout.connect(when_ready)
        poll.start(30)
    window.closed.connect(application.quit)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
