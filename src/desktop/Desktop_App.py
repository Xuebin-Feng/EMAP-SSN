# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Consolidated Desktop Qt application presentation, windowing, fonts, layouts and translations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import TYPE_CHECKING
import warnings
import weakref

from PySide6 import QtCore, QtNetwork
from PySide6.QtCore import QEvent, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QFont, QFontMetrics, QPalette
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStyle,
    QStyleOptionButton,
    QTabWidget,
    QTextEdit,
)

from utilities.App_Settings import AppSettingsError, read_app_settings, save_app_setting

from utilities.Localization import (
    CATALOG_NAME,
    LANGUAGES_DIR,
    MESSAGE_CONTEXT,
    QT_TRANSLATE_NOOP,
    english_plural,
    pseudo_translate,
    read_catalog,
    set_translator,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

# =====================================================================
# 1. Application Identity & Linux Desktop Configuration
# =====================================================================

PRODUCT_NAME = "EMAP-SSN"
APPLICATION_VERSION = "0.3.0"
PRODUCT_LONG_NAME = (
    f"{PRODUCT_NAME}: Embedding- and Multiple-Alignment-integrated Protein "
    "Sequence Similarity Network Platform"
)
# Each window's name, in English. A window shows it translated under its
# own context: QCoreApplication.translate("Config", CONFIG_DISPLAY_NAME).
CONFIG_DISPLAY_NAME = QT_TRANSLATE_NOOP("Config", "EMAP-SSN Configuration")
VIEWER_DISPLAY_NAME = QT_TRANSLATE_NOOP("Viewer", "EMAP-SSN Viewer")
TOOLS_DISPLAY_NAME = QT_TRANSLATE_NOOP("Tools", "EMAP-SSN Tools")

TOOLS_DESKTOP_FILE_NAME = "emapssn_tools"
VIEWER_DESKTOP_FILE_NAME = "emapssn"


def configure_linux_qt_desktop_identity(application, desktop_file_name):
    """Let Linux desktops associate a Qt window with its ``.desktop`` file."""
    if not sys.platform.startswith("linux"):
        return

    # Qt expects the desktop-entry basename without the trailing extension.
    # The application name also supplies the X11 WM_CLASS used by older
    # desktops and by XWayland, which the launchers prefer on Wayland sessions.
    application.setApplicationName(desktop_file_name)
    application.setDesktopFileName(desktop_file_name)


# =====================================================================
# 2. Main Window Focus, Single-Instance Channel & OS Window Management
# =====================================================================

def _single_instance_server_name(application_id):
    """Return a short, stable, per-user local-server name for an application."""
    normalized_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(application_id)).strip("-")
    if not normalized_id:
        raise ValueError("application_id must contain at least one usable character")
    user_identity = os.path.normcase(str(Path.home().resolve()))
    user_digest = hashlib.sha256(user_identity.encode("utf-8")).hexdigest()[:12]
    return f"ssn-{normalized_id}-{user_digest}"


def _notify_local_server(server_name, timeout_ms):
    socket = QtNetwork.QLocalSocket()
    socket.connectToServer(server_name)
    if not socket.waitForConnected(max(0, int(timeout_ms))):
        socket.abort()
        return False
    socket.write(b"activate\n")
    socket.flush()
    socket.waitForBytesWritten(min(max(0, int(timeout_ms)), 500))
    socket.disconnectFromServer()
    return True


def notify_existing_instance(application_id, timeout_ms=750):
    """Activate an existing application without claiming a new instance."""
    return _notify_local_server(
        _single_instance_server_name(application_id), timeout_ms
    )


class SingleInstanceController(QtCore.QObject):
    """Own one application instance and route duplicate launches to its window."""

    def __init__(self, application_id, parent=None):
        super().__init__(parent)
        self.server_name = _single_instance_server_name(application_id)
        lock_path = Path(QtCore.QDir.tempPath()) / f"{self.server_name}.lock"
        self._lock = QtCore.QLockFile(str(lock_path))
        self._server = QtNetwork.QLocalServer(self)
        self._server.setSocketOptions(QtNetwork.QLocalServer.UserAccessOption)
        self._server.newConnection.connect(self._accept_connections)
        self._activation_callback = None
        self._activation_pending = False
        self._owns_server = False

    @property
    def owns_server(self):
        return self._owns_server

    def acquire_or_notify(self, timeout_ms=30000):
        """Claim the instance name, or ask the owner to activate and return False."""
        if not self._lock.tryLock(0):
            deadline = time.monotonic() + (max(0, timeout_ms) / 1000.0)
            while True:
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                if self._notify_existing_instance(min(remaining_ms, 500)):
                    return False
                if remaining_ms <= 0:
                    break
                QtCore.QThread.msleep(min(100, remaining_ms))
            raise RuntimeError(
                "Another instance is running, but its window could not be activated."
            )

        if self._server.listen(self.server_name):
            self._owns_server = True
            return True
        QtNetwork.QLocalServer.removeServer(self.server_name)
        if self._server.listen(self.server_name):
            self._owns_server = True
            return True
        self._lock.unlock()
        raise RuntimeError(
            f"Could not establish the single-instance channel "
            f"'{self.server_name}': {self._server.errorString()}"
        )

    def _notify_existing_instance(self, timeout_ms):
        return _notify_local_server(self.server_name, timeout_ms)

    def set_activation_callback(self, callback):
        """Register the callback used to restore and focus the primary window."""
        if callback is not None and not callable(callback):
            raise TypeError("activation callback must be callable or None")
        self._activation_callback = callback
        if callback is not None and self._activation_pending:
            self._activation_pending = False
            QtCore.QTimer.singleShot(0, callback)

    def _accept_connections(self):
        accepted = False
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            if socket is None:
                break
            accepted = True
            socket.disconnectFromServer()
            socket.deleteLater()
        if not accepted:
            return
        if self._activation_callback is None:
            self._activation_pending = True
        else:
            QtCore.QTimer.singleShot(0, self._activation_callback)

    def close(self):
        """Release the owned local server during normal application shutdown."""
        if self._owns_server:
            self._server.close()
            self._owns_server = False
            self._lock.unlock()


def _activate_window(window):
    """Request foreground focus without leaving the window always-on-top."""
    if window is None or not window.isVisible():
        return

    if window.isMinimized():
        window.showNormal()
    window.raise_()
    window.activateWindow()

    handle = window.windowHandle()
    if handle is not None:
        handle.requestActivate()

    if sys.platform == "win32":
        _activate_windows_native(window)


def _activate_windows_native(window):
    """Put a Windows main window in front once, then restore normal Z-order."""
    try:
        import ctypes

        hwnd = int(window.winId())
        user32 = ctypes.windll.user32
        swp_no_move_or_size = 0x0001 | 0x0002
        hwnd_topmost = -1
        hwnd_not_topmost = -2
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetWindowPos(
            hwnd, hwnd_topmost, 0, 0, 0, 0, swp_no_move_or_size
        )
        user32.SetWindowPos(
            hwnd, hwnd_not_topmost, 0, 0, 0, 0, swp_no_move_or_size
        )
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    except (AttributeError, OSError, TypeError, ValueError):
        pass


def _signal_launcher_ready(window):
    """Atomically tell a desktop launcher that its Qt window is ready."""
    if window is None or not window.isVisible():
        return False

    ready_value = os.environ.pop("SSN_GUI_READY_FILE", None)
    if not ready_value:
        return False

    ready_path = Path(ready_value)
    temporary_path = ready_path.with_name(
        f".{ready_path.name}.{os.getpid()}.tmp"
    )
    try:
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text("ready\n", encoding="utf-8")
        os.replace(temporary_path, ready_path)
    except OSError as error:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        print(f"Warning: Could not signal GUI readiness: {error}", file=sys.stderr)
        return False
    return True


def _activate_and_signal(window):
    """Perform the final foreground request, then acknowledge GUI readiness."""
    _activate_window(window)
    _signal_launcher_ready(window)


def show_window_in_front(window):
    """Show a main window and make best-effort foreground requests at startup."""
    window.show()
    QtCore.QTimer.singleShot(
        0, lambda active_window=window: _activate_window(active_window)
    )
    QtCore.QTimer.singleShot(
        100, lambda active_window=window: _activate_and_signal(active_window)
    )


def open_in_file_manager(path):
    """Reveal a path through Qt, with a platform shell fallback."""
    target = os.path.abspath(path)
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        if QDesktopServices.openUrl(QUrl.fromLocalFile(target)):
            return True
    except Exception:
        pass

    try:
        import subprocess

        if os.name == "nt":
            os.startfile(target)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
        return True
    except Exception:
        return False


# =====================================================================
# 3. Typography, Noto Fonts & VisPy Face Registration
# =====================================================================

QT_UI_FAMILY = "Noto Sans"
QT_MONOSPACE_FAMILY = "Noto Sans Mono"
# Simplified Chinese, which the core faces lack. It is registered only while
# that language shows (LANGUAGE_FONTS), and the stacks skip it otherwise.
QT_SIMPLIFIED_CHINESE_FAMILY = "Noto Sans SC"
VISPY_UI_FACE = "NotoSans"
VISPY_MONOSPACE_FACE = "NotoSansMono"
VISPY_SIMPLIFIED_CHINESE_FACE = "NotoSansSC"
VISPY_FALLBACK_FACE = "OpenSans"
VISPY_REFERENCE_DPI = 96.0

QT_UI_FAMILIES = (
    QT_UI_FAMILY,
    QT_SIMPLIFIED_CHINESE_FAMILY,
    "Segoe UI",
    ".AppleSystemUIFont",
    "Helvetica Neue",
    "Cantarell",
    "DejaVu Sans",
    "sans-serif",
)
QT_MONOSPACE_FAMILIES = (
    QT_MONOSPACE_FAMILY,
    QT_SIMPLIFIED_CHINESE_FAMILY,
    "SFMono-Regular",
    "Menlo",
    "Monaco",
    "Consolas",
    "Liberation Mono",
    "DejaVu Sans Mono",
    "monospace",
)


def _qss_stack(families: tuple[str, ...]) -> str:
    return ", ".join(
        family if family in {"sans-serif", "monospace"} else f"'{family}'"
        for family in families
    )


UI_QSS_FONT_STACK = _qss_stack(QT_UI_FAMILIES)
MONOSPACE_QSS_FONT_STACK = _qss_stack(QT_MONOSPACE_FAMILIES)

DESKTOP_FONT_DIR = (
    Path(__file__).resolve().parents[1] / "resources" / "fonts" / "desktop"
)
FONT_MANIFEST = DESKTOP_FONT_DIR / "SHA256SUMS"
NOTO_FONT_DIR = DESKTOP_FONT_DIR / "noto"


def _manifest_entries(manifest_path: Path) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    try:
        lines = manifest_path.read_text(encoding="ascii").splitlines()
    except OSError:
        return ()
    for line in lines:
        checksum, separator, relative_path = line.partition("  ")
        if separator and len(checksum) == 64 and relative_path:
            entries.append((relative_path, checksum.lower()))
    return tuple(entries)


def _font_files_for_directory(font_dir: Path) -> tuple[str, ...]:
    manifest_path = font_dir / "SHA256SUMS"
    if manifest_path == FONT_MANIFEST or not font_dir.exists():
        return FONT_FILES
    entries = _manifest_entries(manifest_path)
    return tuple(
        relative_path
        for relative_path, _ in entries
        if relative_path not in LANGUAGE_FONT_FILES
    )


FONT_MANIFEST_ENTRIES = _manifest_entries(FONT_MANIFEST)

UI_REGULAR_FILE = "noto/NotoSans/NotoSans-Regular.ttf"
UI_BOLD_FILE = "noto/NotoSans/NotoSans-Bold.ttf"
MONOSPACE_REGULAR_FILE = "noto/NotoSansMono/NotoSansMono-Regular.ttf"
MONOSPACE_BOLD_FILE = "noto/NotoSansMono/NotoSansMono-Bold.ttf"


@dataclass(frozen=True)
class LanguageFont:
    """A bundled font for a language whose script the core faces lack.

    Qt finds the family after the core family in each stack, so it shows
    only the characters the core lacks. VisPy draws a text in a single face,
    so the Viewer draws all its text in vispy_face, which has Latin letters
    too. files holds the regular and bold faces, as the manifest names them.
    web_range is the CSS unicode-range of the script's characters, which
    the browser views draw in it (language_web_font_css).
    """

    family: str
    vispy_face: str
    files: tuple[str, ...]
    web_range: str = ""


# Registered only while their language shows, so other languages keep the
# system's fonts for the same characters, as Japanese does for kanji.
LANGUAGE_FONTS = {
    "zh_CN": LanguageFont(
        QT_SIMPLIFIED_CHINESE_FAMILY,
        VISPY_SIMPLIFIED_CHINESE_FACE,
        (
            "noto/NotoSansSC/NotoSansSC-Regular.ttf",
            "noto/NotoSansSC/NotoSansSC-Bold.ttf",
        ),
        # CJK punctuation, kana, bopomofo, enclosed CJK, ideographs and
        # full-width forms; Latin, Greek and Cyrillic keep the core faces.
        "U+3000-303F, U+3040-30FF, U+3100-312F, U+3200-33FF, U+4E00-9FFF, U+F900-FAFF, U+FF00-FFEF",
    ),
}
LANGUAGE_FONT_FILES = frozenset(
    relative_path for font in LANGUAGE_FONTS.values() for relative_path in font.files
)
# The core faces, registered at startup: every bundled file but the language fonts.
FONT_FILES = tuple(
    relative_path
    for relative_path, _ in FONT_MANIFEST_ENTRIES
    if relative_path not in LANGUAGE_FONT_FILES
)


def vispy_points_for_logical_pixels(logical_pixels: float, canvas_dpi: float) -> float:
    """Convert a platform-independent logical-pixel size to VisPy points."""
    logical_pixels = max(0.0, float(logical_pixels))
    try:
        dpi = float(canvas_dpi)
    except (TypeError, ValueError):
        dpi = VISPY_REFERENCE_DPI
    if not math.isfinite(dpi) or dpi <= 0.0:
        dpi = VISPY_REFERENCE_DPI
    return logical_pixels * 72.0 / dpi


def vispy_points_at_reference_dpi(point_size: float, canvas_dpi: float) -> float:
    """Preserve an existing point-size setting's appearance at reference DPI."""
    logical_pixels = max(0.0, float(point_size)) * VISPY_REFERENCE_DPI / 72.0
    return vispy_points_for_logical_pixels(logical_pixels, canvas_dpi)


@dataclass(frozen=True)
class QtFontLoadStatus:
    loaded_files: tuple[str, ...]
    failed_files: tuple[str, ...]
    loaded_families: tuple[str, ...]
    ui_family_available: bool
    monospace_family_available: bool


@dataclass(frozen=True)
class VispyFontLoadStatus:
    ui_face: str
    monospace_face: str
    failed_faces: tuple[str, ...]


_qt_font_ids: dict[Path, int] = {}
_qt_app_ref: weakref.ReferenceType | None = None
_warned_messages: set[str] = set()
_vispy_status_by_dir: dict[Path, VispyFontLoadStatus] = {}


def _warn_once(message: str) -> None:
    if message not in _warned_messages:
        _warned_messages.add(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def configure_qt_application_fonts(
    app: "QApplication", font_dir: str | Path = DESKTOP_FONT_DIR
) -> QtFontLoadStatus:
    """Register bundled Noto fonts and preserve the platform's font metrics."""
    from PySide6.QtGui import QFont, QFontDatabase

    global _qt_app_ref
    cached_app = _qt_app_ref() if _qt_app_ref is not None else None
    if cached_app is not app:
        _qt_font_ids.clear()
        _qt_app_ref = weakref.ref(app)

    resolved_dir = Path(font_dir).resolve()
    font_files = _font_files_for_directory(resolved_dir)
    loaded_files: list[str] = []
    failed_files: list[str] = []
    loaded_families: set[str] = set()

    if not font_files:
        _warn_once(
            "Bundled Noto font manifest is unavailable or invalid: "
            f"{resolved_dir / 'SHA256SUMS'}."
        )
    elif not resolved_dir.is_dir():
        failed_files.extend(font_files)
        _warn_once(f"Bundled Noto font directory is unavailable: {resolved_dir}.")
    else:
        # Removing an application font (a language's, when the language
        # changes) leaves Qt listing no family for any of them until its font
        # database is read again; reading it now registers the rest anew.
        QFontDatabase.families()
        for relative_path in font_files:
            font_path = resolved_dir / Path(relative_path)
            font_id = _qt_font_ids.get(font_path)
            if font_id is None:
                font_id = (
                    QFontDatabase.addApplicationFont(str(font_path))
                    if font_path.is_file()
                    else -1
                )
                _qt_font_ids[font_path] = font_id

            families = (
                tuple(QFontDatabase.applicationFontFamilies(font_id))
                if font_id >= 0
                else ()
            )
            if not families:
                failed_files.append(relative_path)
                _warn_once(f"Bundled font could not be registered: {font_path}.")
                continue

            loaded_files.append(relative_path)
            loaded_families.update(families)

    loaded_file_set = set(loaded_files)
    ui_available = (
        UI_REGULAR_FILE in loaded_file_set and QT_UI_FAMILY in loaded_families
    )
    monospace_available = (
        MONOSPACE_REGULAR_FILE in loaded_file_set
        and QT_MONOSPACE_FAMILY in loaded_families
    )
    if ui_available:
        application_font = QFont(app.font())
        application_font.setFamilies(list(QT_UI_FAMILIES))
        app.setFont(application_font)

    return QtFontLoadStatus(
        loaded_files=tuple(loaded_files),
        failed_files=tuple(failed_files),
        loaded_families=tuple(sorted(loaded_families)),
        ui_family_available=ui_available,
        monospace_family_available=monospace_available,
    )


def force_light_palette(app):
    """Apply the shared light Fusion palette to a Qt application."""
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 233, 233))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(0, 0, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(48, 140, 198))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)


def qt_monospace_font(base_font: "QFont | None" = None) -> "QFont":
    """Return a metric-preserving QFont that prefers bundled Noto Sans Mono."""
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    if base_font is None:
        app = QApplication.instance()
        base_font = app.font() if app is not None else QFont()
    font = QFont(base_font)
    font.setFamilies(list(QT_MONOSPACE_FAMILIES))
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def register_vispy_application_fonts(
    qt_status: QtFontLoadStatus,
    font_dir: str | Path = DESKTOP_FONT_DIR,
) -> VispyFontLoadStatus:
    """Register the core Noto Sans faces supported by VisPy's single-face text."""
    from vispy.util.fonts import register_vispy_font

    resolved_dir = Path(font_dir).resolve()
    cached_status = _vispy_status_by_dir.get(resolved_dir)
    if cached_status is not None:
        return cached_status

    loaded_files = set(qt_status.loaded_files)
    failed_faces: list[str] = []
    ui_ready = {UI_REGULAR_FILE, UI_BOLD_FILE}.issubset(loaded_files)
    mono_ready = {MONOSPACE_REGULAR_FILE, MONOSPACE_BOLD_FILE}.issubset(
        loaded_files
    )

    if ui_ready:
        register_vispy_font(
            str(resolved_dir / "noto" / "NotoSans"),
            VISPY_UI_FACE,
            False,
            False,
        )
    else:
        failed_faces.append(VISPY_UI_FACE)
        _warn_once(
            "Bundled Noto Sans regular/bold faces are incomplete; "
            "VisPy will retain OpenSans."
        )

    if mono_ready:
        register_vispy_font(
            str(resolved_dir / "noto" / "NotoSansMono"),
            VISPY_MONOSPACE_FACE,
            False,
            False,
        )
    else:
        failed_faces.append(VISPY_MONOSPACE_FACE)
        _warn_once(
            "Bundled Noto Sans Mono regular/bold faces are incomplete; "
            "the VisPy console will retain OpenSans."
        )

    status = VispyFontLoadStatus(
        ui_face=VISPY_UI_FACE if ui_ready else VISPY_FALLBACK_FACE,
        monospace_face=(
            VISPY_MONOSPACE_FACE if mono_ready else VISPY_FALLBACK_FACE
        ),
        failed_faces=tuple(failed_faces),
    )
    _vispy_status_by_dir[resolved_dir] = status
    return status


def _add_language_font(
    language: str | None, font_dir: str | Path = DESKTOP_FONT_DIR
) -> tuple[int, ...]:
    """Register the bundled font language needs with Qt, if any.

    Returns the ids that QFontDatabase.removeApplicationFont takes it out by.
    """
    from PySide6.QtGui import QFontDatabase

    font = LANGUAGE_FONTS.get(language)
    if font is None:
        return ()
    font_ids: list[int] = []
    for relative_path in font.files:
        font_path = Path(font_dir).resolve() / relative_path
        font_id = (
            QFontDatabase.addApplicationFont(str(font_path))
            if font_path.is_file()
            else -1
        )
        if font_id < 0:
            _warn_once(f"Bundled font could not be registered: {font_path}.")
        else:
            font_ids.append(font_id)
    return tuple(font_ids)


def vispy_language_face(
    language: str | None, font_dir: str | Path = DESKTOP_FONT_DIR
) -> str | None:
    """Register and return the VisPy face that draws language.

    None means the core faces draw it. A language font missing its regular
    or bold file leaves VisPy with the core faces, with a warning.
    """
    font = LANGUAGE_FONTS.get(language)
    if font is None:
        return None
    resolved_dir = Path(font_dir).resolve()
    if not all((resolved_dir / path).is_file() for path in font.files):
        _warn_once(
            f"Bundled {font.family} regular/bold faces are incomplete in "
            f"{resolved_dir}; VisPy will retain the core faces."
        )
        return None
    from vispy.util.fonts import register_vispy_font

    register_vispy_font(
        str((resolved_dir / font.files[0]).parent), font.vispy_face, False, False
    )
    return font.vispy_face


def matplotlib_language_families(
    language: str | None, font_dir: str | Path = DESKTOP_FONT_DIR
) -> list[str] | None:
    """Register language's bundled font with matplotlib; return the families for its text.

    A figure draws text in these families, matplotlib's own sans-serif
    first, so the bundled font supplies only the characters it lacks.
    Passing them to each text, rather than changing matplotlib's settings,
    leaves every other figure as it was. None means matplotlib's own fonts
    draw the language. A missing font file leaves them, with a warning.
    """
    font = LANGUAGE_FONTS.get(language)
    if font is None:
        return None
    paths = [Path(font_dir).resolve() / relative_path for relative_path in font.files]
    if not all(path.is_file() for path in paths):
        _warn_once(
            f"Bundled {font.family} faces are incomplete in {Path(font_dir).resolve()}; "
            "figures keep matplotlib's fonts."
        )
        return None
    from matplotlib import font_manager

    registered = {Path(entry.fname).resolve() for entry in font_manager.fontManager.ttflist}
    for path in paths:
        if path not in registered:
            font_manager.fontManager.addfont(str(path))
    return ["sans-serif", font.family]


def language_web_font_css(language: str | None, url_prefix: str) -> str:
    """@font-face rules that let a browser view draw language in its bundled font, or "".

    They declare the language font's family, with its regular (400) and
    bold (700) faces, for its script's characters (LanguageFont.web_range).
    The browser views' font stacks name that family after the core one, as
    the Qt stacks do, so it draws only what the core lacks. Added only for
    this language, the rules leave other languages with the system's fonts.
    url_prefix leads to the desktop font folder: "/fonts/desktop/" on the
    Viewer's web server, or a path relative to a page's base URL.
    """
    font = LANGUAGE_FONTS.get(language)
    if font is None or not font.web_range:
        return ""
    return "\n".join(
        f"@font-face {{ font-family: '{font.family}'; font-style: normal; font-weight: {weight}; "
        f"font-display: swap; src: url('{url_prefix}{path}') format('truetype'); "
        f"unicode-range: {font.web_range}; }}"
        for weight, path in zip((400, 700), font.files)
    )


# =====================================================================
# 4. Responsive Layouts (Height-for-Width & Wrapping Flows)
# =====================================================================

class ResponsiveFieldLayout(QLayout):
    """Lay out existing label/control pairs horizontally, or as readable rows."""

    def __init__(self, parent, pairs, ratios, *, field_ratios=False,
                 trailing=False, spacing=30, column_spacing=None,
                 equal_fields=False, control_stretches=None, wrap_labels=True,
                 column_width_provider=None):
        super().__init__(parent)
        self.wrap_labels = wrap_labels
        self.pairs = pairs
        self.ratios = ratios
        self.field_ratios = field_ratios
        self.trailing = trailing
        self.column_spacing = spacing if column_spacing is None else column_spacing
        self.equal_fields = equal_fields
        self.control_stretches = control_stretches
        self.column_width_provider = column_width_provider
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)
        if parent is not None:
            parent.installEventFilter(self)
        for label, control in pairs:
            self.addWidget(label)
            self.addWidget(control)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < self.count() else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < self.count() else None

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.LayoutRequest:
            watched.updateGeometry()
        return super().eventFilter(watched, event)

    def expandingDirections(self):
        return Qt.Orientation.Horizontal

    def hasHeightForWidth(self):
        return True

    @staticmethod
    def _minimum(widget):
        return widget.minimumSizeHint().expandedTo(widget.minimumSize()).boundedTo(
            widget.maximumSize()
        )

    def minimumSize(self):
        width = max(self._minimum(w).width() for pair in self.pairs for w in pair)
        if not self.wrap_labels:
            width = (max(self._label_width(label) for label, _ in self.pairs)
                     + self.spacing()
                     + max(self._minimum(control).width() for _, control in self.pairs))
        return QSize(width, max(self._minimum(w).height()
                               for pair in self.pairs for w in pair))

    def sizeHint(self):
        width = self._wide_width()
        return QSize(width, self.heightForWidth(width))

    def _label_width(self, label):
        return max(label.minimumWidth(), label.sizeHint().width())

    def _column_minima(self):
        gap = self.spacing()
        return [self._label_width(label) + gap + self._minimum(control).width()
                for label, control in self.pairs]

    def _wide_width(self):
        if self.column_width_provider is not None:
            return self.column_width_provider(None)
        minima = self._column_minima()
        gap = self.spacing()
        if self.equal_fields:
            control_width = max(self._minimum(control).width() for _, control in self.pairs)
            return (sum(self._label_width(label) + gap + control_width
                        for label, _ in self.pairs)
                    + self.column_spacing * (len(minima) - 1))
        if self.trailing:
            return sum(minima) + self.column_spacing * (len(minima) - 1)
        prefix = 0
        if self.field_ratios:
            prefix = self._label_width(self.pairs[0][0]) + gap
            minima[0] -= prefix
        unit = max(math.ceil(width / ratio)
                   for width, ratio in zip(minima, self.ratios))
        return prefix + unit * sum(self.ratios) + self.column_spacing * (len(minima) - 1)

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, max(1, width), 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)
        parent = self.parentWidget()
        height = self.heightForWidth(rect.width())
        if parent is not None and getattr(self, "_last_height", None) != height:
            self._last_height = height
            QTimer.singleShot(0, parent.updateGeometry)

    def _arrange(self, rect, *, apply):
        gap = self.spacing()
        row_gap = 12
        wide = rect.width() >= self._wide_width()
        if apply:
            self.parentWidget().setProperty("stacked", not wide)
        shared_label = max(self._label_width(label) for label, _ in self.pairs)

        def place_pair(index, x, y, width, label_width):
            label_item, control_item = self._items[2 * index:2 * index + 2]
            control_min = self._minimum(self.pairs[index][1]).width()
            stacked_label = self.wrap_labels and label_width + gap + control_min > width
            input_width = width if stacked_label else width - label_width - gap
            label_height = label_item.sizeHint().height()
            control_height = (control_item.heightForWidth(input_width)
                              if control_item.hasHeightForWidth()
                              else control_item.sizeHint().height())
            height = (label_height + row_gap + control_height if stacked_label
                      else max(label_height, control_height))
            if apply:
                label_item.setGeometry(QRect(
                    x, y, label_width, label_height if stacked_label else height,
                ))
                control_item.setGeometry(QRect(
                    x if stacked_label else x + label_width + gap,
                    y + label_height + row_gap if stacked_label
                    else y + (height - control_height) // 2,
                    input_width, control_height,
                ))
            return height

        if not wide:
            y = rect.y()
            for index in range(len(self.pairs)):
                y += place_pair(index, rect.x(), y, rect.width(), shared_label)
                y += row_gap
            return y - rect.y() - row_gap

        widths = self._column_minima()
        available = rect.width() - self.column_spacing * (len(widths) - 1)
        if self.column_width_provider is not None:
            widths = self.column_width_provider(rect.width())
        elif self.equal_fields:
            input_space = available - sum(self._label_width(label) + gap
                                          for label, _ in self.pairs)
            widths = []
            for index, (label, _) in enumerate(self.pairs):
                share = input_space // (len(self.pairs) - index)
                widths.append(self._label_width(label) + gap + share)
                input_space -= share
        elif self.trailing:
            flexible = [i for i, (_, control) in enumerate(self.pairs)
                        if not isinstance(control, QPushButton)]
            weights = (self.control_stretches if self.control_stretches is not None
                       else tuple(int(i in flexible) for i in range(len(widths))))
            flexible = [i for i, weight in enumerate(weights) if weight > 0]
            extra = available - sum(widths)
            remaining_weight = sum(weights)
            for index in flexible:
                share = extra * weights[index] // remaining_weight
                remaining_weight -= weights[index]
                widths[index] += share
                extra -= share
        else:
            prefix = (self._label_width(self.pairs[0][0]) + gap
                      if self.field_ratios else 0)
            remaining = available - prefix
            ratio_sum = sum(self.ratios)
            widths = []
            for ratio in self.ratios:
                width = remaining * ratio // ratio_sum
                widths.append(width)
                remaining -= width
                ratio_sum -= ratio
            widths[0] += prefix
        x = rect.x()
        height = 0
        for index, ((label, _), width) in enumerate(zip(self.pairs, widths)):
            height = max(height, place_pair(
                index, x, rect.y(), width, self._label_width(label)
            ))
            x += width + self.column_spacing
        return height


class ResponsiveFlowLayout(QLayout):
    """Wrap visible controls in order, retaining their minimum usable sizes."""

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self._stretches = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)
        if parent is not None:
            parent.installEventFilter(self)

    def addItem(self, item):
        self._items.append(item)
        self._stretches.append(0)

    def addWidget(self, widget, stretch=0):
        super().addWidget(widget)
        self._stretches[-1] = stretch

    def stretch(self, index):
        return self._stretches[index]

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < self.count() else None

    def takeAt(self, index):
        if 0 <= index < self.count():
            self._stretches.pop(index)
            return self._items.pop(index)
        return None

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.LayoutRequest:
            watched.updateGeometry()
        return super().eventFilter(watched, event)

    def expandingDirections(self):
        return Qt.Orientation.Horizontal

    def hasHeightForWidth(self):
        return True

    def _visible(self):
        return [i for i, item in enumerate(self._items) if not item.isEmpty()]

    def _minimum(self, index):
        widget = self._items[index].widget()
        return widget.minimumSizeHint().expandedTo(widget.minimumSize()).boundedTo(
            widget.maximumSize()
        )

    def minimumSize(self):
        sizes = [self._minimum(i) for i in self._visible()]
        return QSize(max((s.width() for s in sizes), default=0),
                     max((s.height() for s in sizes), default=0))

    def sizeHint(self):
        visible = self._visible()
        width = (sum(self._minimum(i).width() for i in visible)
                 + self.spacing() * max(0, len(visible) - 1))
        return QSize(width, self.heightForWidth(width))

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, max(1, width), 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)
        parent = self.parentWidget()
        height = self.heightForWidth(rect.width())
        if parent is not None and getattr(self, "_last_height", None) != height:
            self._last_height = height
            parent.setMinimumHeight(height)
            QTimer.singleShot(0, parent.updateGeometry)

    def _arrange(self, rect, *, apply):
        rows = []
        row = []
        used = 0
        for index in self._visible():
            width = self._minimum(index).width()
            if row and used + self.spacing() + width > rect.width():
                rows.append(row)
                row = []
                used = 0
            used += (self.spacing() if row else 0) + width
            row.append(index)
        if row:
            rows.append(row)
        y = rect.y()
        for row in rows:
            widths = [self._minimum(i).width() for i in row]
            extra = max(0, rect.width() - sum(widths) - self.spacing() * (len(row) - 1))
            weight = sum(self._stretches[i] for i in row)
            for pos, index in enumerate(row):
                if self._stretches[index] and weight:
                    share = extra * self._stretches[index] // weight
                    widths[pos] += share
                    extra -= share
                    weight -= self._stretches[index]
            height = max(self._items[i].sizeHint().height() for i in row)
            x = rect.x()
            for index, width in zip(row, widths):
                item = self._items[index]
                if apply:
                    h = item.sizeHint().height()
                    item.setGeometry(QRect(x, y + (height - h) // 2, width, h))
                x += width + self.spacing()
            y += height + self.spacing()
        return max(0, y - rect.y() - self.spacing())


class ResponsiveSelectorLayout(ResponsiveFlowLayout):
    """Selector, optional name, folder button; keep the folder beside the selector."""

    def sizeHint(self):
        width = self._minimum(0).width()
        if 1 in self._visible():
            width = 2 * max(width, self._minimum(1).width()) + self.spacing()
        width += self.spacing() + self._minimum(2).width()
        return QSize(width, self.heightForWidth(width))

    def minimumSize(self):
        selector_width = self._minimum(0).width() + self.spacing() + self._minimum(2).width()
        name_width = self._minimum(1).width() if 1 in self._visible() else 0
        return QSize(max(selector_width, name_width), super().minimumSize().height())

    def _arrange(self, rect, *, apply):
        visible = self._visible()
        if not visible:
            return 0
        has_name = 1 in visible
        folder_width = self._minimum(2).width()
        gap = self.spacing()
        available = rect.width() - folder_width - gap
        split = has_name and (available - gap) // 2 < max(
            self._minimum(0).width(), self._minimum(1).width()
        )
        height = max(self._items[i].sizeHint().height() for i in (0, 2))
        selector_width = (available - gap) // 2 if has_name and not split else available
        if apply:
            self._items[0].setGeometry(QRect(rect.x(), rect.y(), selector_width, height))
            self._items[2].setGeometry(QRect(rect.right() - folder_width + 1,
                                            rect.y(), folder_width, height))
        if has_name:
            name_height = self._items[1].sizeHint().height()
            if apply:
                self._items[1].setGeometry(QRect(
                    rect.x() if split else rect.x() + selector_width + gap,
                    rect.y() + height + gap if split else rect.y(),
                    rect.width() if split else available - selector_width - gap,
                    name_height,
                ))
            height = height + gap + name_height if split else max(height, name_height)
        return height


class WrappedPlaceholderTextEdit(QTextEdit):
    """A QTextEdit whose placeholder wraps to the box's width.

    Qt shows only the first line of a QTextEdit's placeholder, so a hint
    wider than the box is cut off, in any language. This box leaves Qt's
    placeholder empty and shows the text in a word-wrapped label instead,
    where Qt would draw it, while the document is empty.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._placeholder = QLabel(self.viewport())
        self._placeholder.setObjectName("wrappedPlaceholder")
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._placeholder.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self._placeholder.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._placeholder.hide()
        self.textChanged.connect(self._show_placeholder)

    def placeholderText(self):
        return self._placeholder.text()

    def setPlaceholderText(self, text):
        self._placeholder.setText(text)
        self._show_placeholder()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            # Qt draws a placeholder in the box's font. An inherited font
            # doesn't last: when a box with a stylesheet moves into a layout,
            # Qt resets the label to the application font, so the Config's
            # monospace report font is given to the label outright.
            self._placeholder.setFont(self.font())

    def resizeEvent(self, event):
        # A scroll area gets its viewport's resizes here, scroll bars coming
        # and going included.
        super().resizeEvent(event)
        margin = int(self.document().documentMargin())
        self._placeholder.setGeometry(
            self.viewport().rect().adjusted(margin, margin, -margin, -margin)
        )

    def _show_placeholder(self):
        self._placeholder.setVisible(bool(self._placeholder.text()) and self.document().isEmpty())


# =====================================================================
# 5. Combo Boxes Storing Values Apart from Their Labels
# =====================================================================

def add_combo_options(combo, values, labels=None):
    """Add items that store ``values`` and show ``labels`` (the values by default).

    Code reads and selects items by their stored value (``combo_value``,
    ``select_combo_value``), never by the displayed text, so a label can be
    reworded or translated without changing what settings files contain. The
    ``persistItemData`` property tells the settings collectors to save values.
    """
    labels = values if labels is None else labels
    if len(labels) != len(values):
        raise ValueError("Each combo box value needs exactly one label.")
    for value, label in zip(values, labels):
        combo.addItem(label, value)
    combo.setProperty("persistItemData", True)


def combo_value(combo):
    """Return the selected item's stored value, or "" when nothing is selected."""
    value = combo.currentData()
    return "" if value is None else value


def select_combo_value(combo, value):
    """Select the item storing ``value``; keep the selection if no item does."""
    index = -1 if value is None else combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)
    return index >= 0


# True on an item that shows a name from outside the program, such as a
# device's, which a translation leaves as it is.
NAME_ITEM_ROLE = Qt.ItemDataRole.UserRole + 1


def mark_name_item(combo, index):
    """Mark item index of combo as showing an outside name (NAME_ITEM_ROLE)."""
    combo.setItemData(index, True, NAME_ITEM_ROLE)


# =====================================================================
# 6. Buttons Sized by Their Text
# =====================================================================

# The space between a button's text and each of its two ends. A button sized
# by its text grows with a longer text, such as a translation, instead of
# clipping it. On a toggle switch this is the radius of its rounded ends.
BUTTON_TEXT_PADDING = 14


def shown_button_text(text):
    """A button shows "&&" as "&" and drops the "&" marking a shortcut key."""
    return re.sub("&(.)", r"\1", text)


def text_button_width(texts, font, padding=BUTTON_TEXT_PADDING):
    """Return the width that shows the widest of ``texts`` in ``font``, padded."""
    metrics = QFontMetrics(font)
    widest = max(metrics.horizontalAdvance(shown_button_text(text)) for text in texts)
    return widest + 2 * padding


def clipped_button_text(button):
    """Return how many pixels of the button's text its style has no room for.

    A style can keep a frame or stylesheet padding inside the button, so the
    room left for the text is narrower than the button.
    """
    option = QStyleOptionButton()
    button.initStyleOption(option)
    room = button.style().subElementRect(
        QStyle.SubElement.SE_PushButtonContents, option, button
    ).width()
    text = shown_button_text(button.text())
    return max(0, button.fontMetrics().horizontalAdvance(text) - room)


def fit_buttons_to_text(*buttons, padding=BUTTON_TEXT_PADDING):
    """Give ``buttons`` one fixed width: their widest text plus ``padding`` at each end.

    Should a style keep more room inside a button than the padding, the
    width grows by what it would clip, so no text is ever cut off.
    """
    width = 0
    for button in buttons:
        button.ensurePolished()  # A stylesheet may set the font the text is drawn in.
        width = max(width, text_button_width([button.text()], button.font(), padding))
    for button in buttons:
        button.setFixedWidth(width)
    width += max(clipped_button_text(button) for button in buttons)
    for button in buttons:
        button.setFixedWidth(width)
    return width


TOGGLE_SWITCH_HEIGHT = 28
TOGGLE_ON_STYLESHEET = (
    "QPushButton { background-color: #4CAF50; color: white; border-radius: 14px; "
    "font-weight: bold; border: 1px solid #388E3C; }"
)
TOGGLE_OFF_STYLESHEET = (
    "QPushButton { background-color: #e0e0e0; color: #333; border-radius: 14px; "
    "font-weight: bold; border: 1px solid #bdbdbd; }"
)


class ToggleSwitch(QPushButton):
    """A checkable pill that reads ``on_text`` or ``off_text``, "ON" and "OFF" by default.

    Its width fits the longer of the two texts, so toggling never resizes it.
    Each toggle swaps the whole stylesheet, as the per-window copies this
    replaces did, so a disabled style the Config's profile gating appends lasts
    until the next toggle.
    """

    def __init__(self, on_text=None, off_text=None, parent=None):
        super().__init__(parent)
        if on_text is None:
            on_text = QtCore.QCoreApplication.translate("ToggleSwitch", "ON")
        if off_text is None:
            off_text = QtCore.QCoreApplication.translate("ToggleSwitch", "OFF")
        self._texts = (off_text, on_text)
        self.setCheckable(True)
        bold = QFont(self.font())
        bold.setBold(True)  # Both stylesheets draw the text bold.
        self.setFixedSize(text_button_width(self._texts, bold), TOGGLE_SWITCH_HEIGHT)
        self.toggled.connect(self._show_state)
        self._show_state(False)

    def state_texts(self):
        """The texts it reads off and on, whichever shows now."""
        return self._texts

    def _show_state(self, checked):
        self.setText(self._texts[bool(checked)])
        self.setStyleSheet(TOGGLE_ON_STYLESHEET if checked else TOGGLE_OFF_STYLESHEET)


# =====================================================================
# 7. Translations: Catalogs, the Pseudo-Language & Startup Loading
# =====================================================================

PSEUDO_LANGUAGE = "pseudo"
PSEUDO_TRANSLATION_VARIABLE = "SSN_PSEUDO_TRANSLATION"
# Qt's own catalogs: "qt" covers its core modules' text (standard buttons,
# context menus, file and colour dialogs), "qtwebengine" the web pages' menus.
QT_CATALOG_PREFIXES = ("qt", "qtwebengine")


def startup_language(environment=None, settings=None, catalog_dir=LANGUAGES_DIR, ui_languages=None):
    """The language windows start in, or None for English as written.

    SSN_PSEUDO_TRANSLATION=1 picks the test-only pseudo-language. Otherwise
    the saved LANGUAGE setting decides, as resolve_language explains;
    settings stands in for app_settings.json, and ui_languages for the
    system's display languages.
    """
    environment = os.environ if environment is None else environment
    value = environment.get(PSEUDO_TRANSLATION_VARIABLE, "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return PSEUDO_LANGUAGE
    return resolve_language(configured_language(settings), catalog_dir, ui_languages)


class CatalogTranslator(QtCore.QTranslator):
    """A translator that answers from a table, and only for the texts in it."""

    def __init__(self, table, parent=None):
        super().__init__(parent)
        self._table = dict(table)

    def translate(self, context, source_text, disambiguation=None, n=-1):
        # None lets Qt ask the next translator, or show the English text.
        return self._table.get((context, source_text, disambiguation or ""))

    def isEmpty(self):
        return not self._table


def pseudo_translator(template):
    """The pseudo-language: every text in the template catalog, pseudo-translated.

    A text missing from the catalog stays English, as it would in a real
    language. So does a text built at run time, such as an f-string passed
    to tr(), which no catalog can list.
    """
    return CatalogTranslator(
        (message.key, pseudo_translate(message.source)) for message in read_catalog(template)
    )


def translate(context, text, disambiguation=None, n=-1):
    """The text a window shows for text: QCoreApplication.translate, with English plurals.

    Windows mark their text with it, as translate("Config", "Save"). A
    counted text (n is the count) that no catalog translates reads as
    English writes it for n: "%n file(s)" shows "1 file" or "2 files"
    (Localization.english_plural), whatever is installed. Pass the
    arguments in order, which lupdate needs to see them.
    """
    shown = QtCore.QCoreApplication.translate(context, text, disambiguation, n)
    return shown if n is None or n < 0 else english_plural(shown, n)


@dataclass
class InstalledTranslations:
    """What install_translations put in place; remove() takes it out again."""

    language: str
    translators: tuple
    previous_message_translator: object = None
    catalog_dir: Path = LANGUAGES_DIR
    font_ids: tuple = ()
    extra_catalogs: tuple = ()

    def remove(self):
        global _installed_translations
        from PySide6.QtGui import QFontDatabase

        for translator in self.translators:
            QtCore.QCoreApplication.removeTranslator(translator)
        set_translator(self.previous_message_translator)
        for font_id in self.font_ids:
            QFontDatabase.removeApplicationFont(font_id)
        if _installed_translations is self:
            _installed_translations = None


# QCoreApplication does not own its translators, so this keeps them alive.
_installed_translations: InstalledTranslations | None = None


def _translate_message(template, n=-1):
    return translate(MESSAGE_CONTEXT, template, None, n)


def _load_catalog(prefix, language, directory):
    """A translator holding <directory>/<prefix>_<language>.qm, or None."""
    translator = QtCore.QTranslator()
    if translator.load(QtCore.QLocale(language), prefix, "_", str(directory)):
        return translator
    return None


def install_translations(app, language, catalog_dir=LANGUAGES_DIR, extra_catalogs=()):
    """Show the windows built after this call in language.

    None or "en" keeps every text English as written and installs nothing.
    PSEUDO_LANGUAGE installs the test-only pseudo-language made from
    <catalog_dir>/emapssn.ts. Any other language loads Qt's own catalogs
    for it, where Qt has them, then <catalog_dir>/emapssn_<language>.qm,
    which wins where both translate a text. Without that catalog it raises
    LookupError and installs nothing. Message texts (the Viewer's console
    line) then come from the same catalogs. A language whose script the
    core fonts lack also registers its bundled font (LANGUAGE_FONTS).

    extra_catalogs lists (directory, name) pairs of catalogs a window adds
    to the main one, as VR Config adds opt_vr's emapssn_vr: the
    pseudo-language is made from <directory>/<name>.ts too, and a language
    loads <directory>/<name>_<language>.qm after the main catalog. One that
    is missing leaves its own texts English.

    Windows set their text once, as they are built, so call this after the
    QApplication exists and before the first window. A second call replaces
    the first. Returns the InstalledTranslations, or None for English.
    """
    global _installed_translations
    if _installed_translations is not None:
        _installed_translations.remove()
    if language in (None, "", "en"):
        return None
    extra_catalogs = tuple((Path(directory), name) for directory, name in extra_catalogs)
    if language == PSEUDO_LANGUAGE:
        translators = [pseudo_translator(Path(catalog_dir) / f"{CATALOG_NAME}.ts")]
        translators += [
            pseudo_translator(directory / f"{name}.ts")
            for directory, name in extra_catalogs
            if (directory / f"{name}.ts").is_file()
        ]
    else:
        ours = _load_catalog(CATALOG_NAME, language, catalog_dir)
        if ours is None:
            raise LookupError(f"No {CATALOG_NAME} catalog for language {language!r} in {catalog_dir}.")
        qt_directory = QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.TranslationsPath)
        translators = [
            translator
            for prefix in QT_CATALOG_PREFIXES
            if (translator := _load_catalog(prefix, language, qt_directory)) is not None
        ]
        translators.append(ours)  # Installed after Qt's, so Qt asks it first.
        translators += [
            translator
            for directory, name in extra_catalogs
            if (translator := _load_catalog(name, language, directory)) is not None
        ]
    for translator in translators:
        app.installTranslator(translator)
    _installed_translations = InstalledTranslations(
        language,
        tuple(translators),
        set_translator(_translate_message),
        Path(catalog_dir),
        _add_language_font(language),
        extra_catalogs,
    )
    return _installed_translations


def installed_language():
    """The language install_translations put in place, or None for English."""
    return None if _installed_translations is None else _installed_translations.language


# =====================================================================
# 8. The Language Option: Setting, Dropdown & Redraw
# =====================================================================

LANGUAGE_SETTING = "LANGUAGE"
SYSTEM_LANGUAGE = "system"
ENGLISH = "en"
# Chinese is written in two scripts, which QLocale tells apart only by country.
_LANGUAGE_NAMES = {ENGLISH: "English", "zh_CN": "简体中文", "zh_TW": "繁體中文"}
_COMPILED_CATALOG = re.compile(rf"^{CATALOG_NAME}_(\w+)\.qm$")


def catalog_languages(catalog_dir=LANGUAGES_DIR):
    """The languages that have a compiled catalog, by code, sorted."""
    codes = []
    for path in sorted(Path(catalog_dir).glob(f"{CATALOG_NAME}_*.qm")):
        match = _COMPILED_CATALOG.match(path.name)
        if match and match.group(1) != ENGLISH:
            codes.append(match.group(1))
    return codes


def language_name(code):
    """A language's name in that language, as the Language dropdown lists it."""
    if code in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[code]
    name = QtCore.QLocale(code).nativeLanguageName()
    return name[:1].upper() + name[1:] if name else code


def system_language(available, ui_languages=None):
    """The first of the system's display languages that EMAP-SSN has, else English.

    ui_languages lists the system's languages in order of preference, as
    QLocale.system().uiLanguages() does.
    """
    if ui_languages is None:
        ui_languages = QtCore.QLocale.system().uiLanguages()
    for tag in ui_languages:
        name = QtCore.QLocale(tag).name()  # language_TERRITORY: zh_CN for zh-Hans-CN
        for code in (name, name.split("_")[0]):
            if code == ENGLISH or code in available:
                return code
    return ENGLISH


def configured_language(settings=None):
    """The saved LANGUAGE setting: SYSTEM_LANGUAGE, ENGLISH or a language code.

    settings stands in for app_settings.json. A file that can't be read
    counts as no setting, with a warning, so the windows follow the system.
    """
    if settings is None:
        try:
            settings = read_app_settings()
        except AppSettingsError as error:
            _warn_once(f"{error} The windows follow the system language.")
            settings = {}
    value = settings.get(LANGUAGE_SETTING)
    return value if isinstance(value, str) and value else SYSTEM_LANGUAGE


def resolve_language(setting, catalog_dir=LANGUAGES_DIR, ui_languages=None):
    """The language a setting shows, as install_translations takes it: None for English.

    SYSTEM_LANGUAGE follows the system's display language where EMAP-SSN
    has a catalog for it. A language whose catalog is gone shows English.
    """
    available = catalog_languages(catalog_dir)
    code = system_language(available, ui_languages) if setting == SYSTEM_LANGUAGE else setting
    if code == ENGLISH:
        return None
    if code not in available:
        _warn_once(f"There is no catalog for the language {code!r}; the windows show English.")
        return None
    return code


class LanguageSelector(QComboBox):
    """The Language dropdown: the system's language, English and every language with a catalog.

    Each language is named in itself, so anyone can find theirs whatever
    language the window shows. The dropdown starts at the saved setting;
    choosing another entry emits language_chosen with the new setting, for
    choose_language to save. show_setting marks a saved choice, and revert
    goes back to it when saving a new one fails.
    """

    language_chosen = QtCore.Signal(str)

    def __init__(self, parent=None, *, catalog_dir=LANGUAGES_DIR, ui_languages=None, setting=None):
        super().__init__(parent)
        self.setObjectName("languageSelector")
        available = catalog_languages(catalog_dir)
        system = language_name(system_language(available, ui_languages))
        self.addItem(
            QtCore.QCoreApplication.translate("LanguageSelector", "System default ({language})")
            .format(language=system),
            SYSTEM_LANGUAGE,
        )
        self.addItem(language_name(ENGLISH), ENGLISH)
        for code in available:
            self.addItem(language_name(code), code)
        self.setToolTip(QtCore.QCoreApplication.translate(
            "LanguageSelector",
            "The language of the windows. Config and Tools switch at once; "
            "other windows use it the next time they open.",
        ))
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.show_setting(configured_language() if setting is None else setting)
        self.currentIndexChanged.connect(self._chosen)

    def show_setting(self, setting):
        """Show setting as the saved choice, without announcing it."""
        index = self.findData(setting)
        self.blockSignals(True)
        try:
            self.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.blockSignals(False)
        self._saved = self.currentData()

    def setting(self):
        """The setting the dropdown shows."""
        return self.currentData()

    def revert(self):
        """Show the saved choice again, as when saving a new one failed."""
        self.show_setting(self._saved)

    def _chosen(self, index):
        self.language_chosen.emit(self.itemData(index))


def choose_language(window, setting):
    """Save setting for every window, then redraw window in the language it names.

    window has a language_selector and a switch_language(language) method.
    If the setting can't be saved, the dropdown goes back to the saved
    choice and a message says why. Returns the window that shows the
    language: the redrawn one, or window itself.
    """
    try:
        save_app_setting(LANGUAGE_SETTING, setting)
    except (AppSettingsError, OSError) as error:
        window.language_selector.revert()
        QMessageBox.critical(
            window,
            QtCore.QCoreApplication.translate("LanguageSelector", "Language Not Saved"),
            QtCore.QCoreApplication.translate(
                "LanguageSelector", "The language could not be saved, so it stays as it was.\n\n{error}"
            ).format(error=error),
        )
        return window
    window.language_selector.show_setting(setting)
    return window.switch_language(resolve_language(setting))


def language_selector_row(selector):
    """A row that ends in the Language dropdown, after a globe."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addStretch()
    globe = QLabel("🌐")
    globe.setObjectName("languageGlobe")
    globe.setBuddy(selector)
    row.addWidget(globe)
    row.addWidget(selector)
    return row


_SIZE_STATES = Qt.WindowState.WindowMaximized | Qt.WindowState.WindowFullScreen


def capture_view(window):
    """What a redraw keeps of a window's view: geometry, splitters, tabs and scroll positions."""
    return {
        # The size and place of a normal window; a maximized one goes back there too.
        "geometry": QRect(window.normalGeometry()),
        "state": window.windowState() & _SIZE_STATES,
        "splitters": [splitter.saveState() for splitter in window.findChildren(QSplitter)],
        "tabs": [tabs.currentIndex() for tabs in window.findChildren(QTabWidget)],
        "scrolls": [
            (area.horizontalScrollBar().value(), area.verticalScrollBar().value())
            for area in window.findChildren(QAbstractScrollArea)
        ],
    }


def _matching(widgets, values):
    """Pairs of widget and value, or none at all when the window was built differently."""
    return list(zip(widgets, values)) if len(widgets) == len(values) else []


def restore_view(window, view):
    """Before showing a window built the same way: its geometry, maximized state and tabs.

    setGeometry rather than restoreGeometry, which would move a window that
    reaches past the screen's edge.
    """
    window.setGeometry(view["geometry"])
    window.setWindowState(view["state"])
    for tabs, index in _matching(window.findChildren(QTabWidget), view["tabs"]):
        tabs.setCurrentIndex(index)


def restore_view_positions(window, view):
    """Once the window is laid out at its size: splitter and scroll positions."""
    for splitter, state in _matching(window.findChildren(QSplitter), view["splitters"]):
        splitter.restoreState(state)
    QApplication.processEvents()
    for area, (horizontal, vertical) in _matching(window.findChildren(QAbstractScrollArea), view["scrolls"]):
        area.horizontalScrollBar().setValue(horizontal)
        area.verticalScrollBar().setValue(vertical)


# Windows made by a redraw; nothing else would keep them alive.
_redrawn_windows = []


def redraw_in_language(window, language, build, catalog_dir=LANGUAGES_DIR, extra_catalogs=()):
    """Replace window with a copy in language, where the window was and as it was.

    Installs language, with the window's extra_catalogs as install_translations
    takes them, then calls build() for the replacement, which must carry over
    the window's values. The replacement gets the window's size, position,
    splitter positions, tabs and scroll positions, and the single-instance
    controller's attention (window.single_instance). It is shown in the
    window's place before the window closes, so the program never runs
    without a window. If build() fails, the window and its language stay.
    Returns the replacement.
    """
    app = QApplication.instance()
    previous = _installed_translations
    view = capture_view(window)
    install_translations(app, language, catalog_dir, extra_catalogs)
    try:
        replacement = build()
    except BaseException:
        if previous is None:
            install_translations(app, None)
        else:
            install_translations(app, previous.language, previous.catalog_dir, previous.extra_catalogs)
        raise
    restore_view(replacement, view)
    replacement.show()
    for _ in range(3):
        app.processEvents()
    restore_view_positions(replacement, view)
    controller = getattr(window, "single_instance", None)
    replacement.single_instance = controller
    if controller is not None:
        controller.set_activation_callback(lambda active=replacement: show_window_in_front(active))
    replacement.raise_()
    replacement.activateWindow()
    _redrawn_windows[:] = [kept for kept in _redrawn_windows if kept is not window] + [replacement]
    window.hide()
    window.close()
    window.deleteLater()
    return replacement


__all__ = [
    "PRODUCT_NAME",
    "APPLICATION_VERSION",
    "PRODUCT_LONG_NAME",
    "CONFIG_DISPLAY_NAME",
    "VIEWER_DISPLAY_NAME",
    "TOOLS_DISPLAY_NAME",
    "TOOLS_DESKTOP_FILE_NAME",
    "VIEWER_DESKTOP_FILE_NAME",
    "configure_linux_qt_desktop_identity",
    "notify_existing_instance",
    "SingleInstanceController",
    "show_window_in_front",
    "open_in_file_manager",
    "QT_UI_FAMILY",
    "QT_MONOSPACE_FAMILY",
    "QT_SIMPLIFIED_CHINESE_FAMILY",
    "VISPY_UI_FACE",
    "VISPY_MONOSPACE_FACE",
    "VISPY_SIMPLIFIED_CHINESE_FACE",
    "VISPY_FALLBACK_FACE",
    "VISPY_REFERENCE_DPI",
    "QT_UI_FAMILIES",
    "QT_MONOSPACE_FAMILIES",
    "UI_QSS_FONT_STACK",
    "MONOSPACE_QSS_FONT_STACK",
    "DESKTOP_FONT_DIR",
    "FONT_MANIFEST",
    "NOTO_FONT_DIR",
    "FONT_MANIFEST_ENTRIES",
    "FONT_FILES",
    "UI_REGULAR_FILE",
    "UI_BOLD_FILE",
    "MONOSPACE_REGULAR_FILE",
    "MONOSPACE_BOLD_FILE",
    "LanguageFont",
    "LANGUAGE_FONTS",
    "LANGUAGE_FONT_FILES",
    "vispy_points_for_logical_pixels",
    "vispy_points_at_reference_dpi",
    "QtFontLoadStatus",
    "VispyFontLoadStatus",
    "configure_qt_application_fonts",
    "force_light_palette",
    "qt_monospace_font",
    "register_vispy_application_fonts",
    "vispy_language_face",
    "matplotlib_language_families",
    "language_web_font_css",
    "ResponsiveFieldLayout",
    "ResponsiveFlowLayout",
    "ResponsiveSelectorLayout",
    "WrappedPlaceholderTextEdit",
    "add_combo_options",
    "combo_value",
    "select_combo_value",
    "NAME_ITEM_ROLE",
    "mark_name_item",
    "BUTTON_TEXT_PADDING",
    "shown_button_text",
    "text_button_width",
    "clipped_button_text",
    "fit_buttons_to_text",
    "TOGGLE_SWITCH_HEIGHT",
    "TOGGLE_ON_STYLESHEET",
    "TOGGLE_OFF_STYLESHEET",
    "ToggleSwitch",
    "PSEUDO_LANGUAGE",
    "PSEUDO_TRANSLATION_VARIABLE",
    "QT_CATALOG_PREFIXES",
    "startup_language",
    "CatalogTranslator",
    "pseudo_translator",
    "translate",
    "InstalledTranslations",
    "install_translations",
    "installed_language",
    "LANGUAGE_SETTING",
    "SYSTEM_LANGUAGE",
    "ENGLISH",
    "catalog_languages",
    "language_name",
    "system_language",
    "configured_language",
    "resolve_language",
    "LanguageSelector",
    "choose_language",
    "language_selector_row",
    "capture_view",
    "restore_view",
    "restore_view_positions",
    "redraw_in_language",
]
