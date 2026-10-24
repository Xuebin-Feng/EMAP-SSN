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
from PySide6.QtCore import QEvent, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QFontMetricsF,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QTextCursor,
    QTextLength,
    QTextTable,
)
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStyle,
    QStyleOptionButton,
    QTabWidget,
    QTextEdit,
    QWidget,
)

from desktop.Studio_Theme import (
    DEFAULT_THEME_COLOR,
    SWITCH_LABEL_GAP,
    SWITCH_STYLESHEET,
    SWITCH_WIDTH,
    THEME_COLORS,
    TOKENS,
    IconLabel,
    SwitchFocusRing,
    apply_studio_theme,
    paint_switch,
    switch_label,
    theme_color,
    theme_tokens,
    widget_class_fonts,
)
from utilities.App_Settings import AppSettingsError, PROJECT_ROOT, read_app_settings, save_app_setting

from utilities.Language_Packs import (
    WEB_LANGUAGE_FAMILY,
    LanguageFont,
    ManifestError,
    find_catalog,
    language_codes,
    read_pack,
    read_packs,
    web_font_css,
)
from utilities.Localization import (
    CATALOG_NAME,
    LANGUAGES_DIR,
    MESSAGE_CONTEXT,
    QT_TRANSLATE_NOOP,
    english_plural,
    english_text,
    pseudo_translate,
    read_catalog,
    set_punctuation,
    set_translator,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

# =====================================================================
# 1. Application Identity & Linux Desktop Configuration
# =====================================================================

PRODUCT_NAME = "EMAP-SSN"
APPLICATION_VERSION = "0.4.0"
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


# The AppUserModelID of each window's process, and of its shortcut (install.bat sets the same IDs on
# the shortcuts it makes, tests/test_launcher_stub.py keeps the two in step): Windows groups a
# process's windows in the taskbar under the shortcut that carries its ID, with the shortcut's name and
# icon, so the taskbar says "EMAP-SSN" and not "Python", and a pinned shortcut finds its window again.
APP_USER_MODEL_IDS = {
    "config": "ca.utoronto.emapssn.config",
    "tools": "ca.utoronto.emapssn.tools",
    "viewer": "ca.utoronto.emapssn.viewer",
    "vr_config": "ca.utoronto.emapssn.vrconfig",
}


def set_app_user_model_id(app_id):
    """Give this process its taskbar identity (Windows only; call it before the first window exists).

    Returns whether Windows accepted it; anywhere else it does nothing and returns False.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        function = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        function.argtypes = [ctypes.c_wchar_p]
        function.restype = ctypes.c_long
        return function(app_id) == 0
    except (AttributeError, OSError):
        return False


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

def _single_instance_server_name(application_id, checkout_root=PROJECT_ROOT):
    """Return a short, stable local-server name for an application, per user and checkout.

    Each checkout keeps its own settings files, so a launch from another
    checkout opens a window of its own: bringing this checkout's window
    forward would have it edit this checkout's settings instead.
    """
    normalized_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(application_id)).strip("-")
    if not normalized_id:
        raise ValueError("application_id must contain at least one usable character")
    identity = "\n".join(
        os.path.normcase(str(Path(path).resolve())) for path in (Path.home(), checkout_root)
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"ssn-{normalized_id}-{digest}"


def instance_handover_message(window_name):
    """The line a launch prints when it brings its checkout's open window forward instead."""
    return (
        f"{window_name} is already open from this checkout ({PROJECT_ROOT}); "
        "brought that window to the front instead of opening another."
    )


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

    # The start is over: no more status for the splash, and no process this window starts may write to it.
    os.environ.pop("SSN_STARTUP_STATUS_FILE", None)
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


def signal_launcher_ready(window):
    """Tell the desktop launcher, if one started this process, that ``window`` is up (it closes its splash)."""
    return _signal_launcher_ready(window)


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
VISPY_UI_FACE = "NotoSans"
VISPY_MONOSPACE_FACE = "NotoSansMono"
VISPY_FALLBACK_FACE = "OpenSans"
VISPY_REFERENCE_DPI = 96.0

# The bundled font of each language whose script the core faces lack, from
# its language.json (utilities/Language_Packs.py). Registered only while its
# language shows, so other languages keep the system's fonts for the same
# characters, as Japanese does for kanji. A language.json that can't be used
# is left out, with a warning when the languages are listed (catalog_languages).
_LANGUAGE_PACKS, _LANGUAGE_PACK_PROBLEMS = read_packs(LANGUAGES_DIR)
LANGUAGE_FONTS = {code: pack.font for code, pack in _LANGUAGE_PACKS.items() if pack.font is not None}
# Each stack names the language fonts after its core family, so a language
# font draws only what the core lacks. A family that isn't registered (its
# language isn't showing) is skipped.
LANGUAGE_FONT_FAMILIES = tuple(dict.fromkeys(font.family for font in LANGUAGE_FONTS.values()))
# WEB_LANGUAGE_FAMILY (utilities/Language_Packs.py) is the family the browser views draw a
# language's script in. language_web_font_css declares the shown language's bundled font under
# it, so a page names one family whatever its language.

_SYSTEM_UI_FAMILIES = (
    "Segoe UI",
    ".AppleSystemUIFont",
    "Helvetica Neue",
    "Cantarell",
    "DejaVu Sans",
    "sans-serif",
)
_SYSTEM_MONOSPACE_FAMILIES = (
    "SFMono-Regular",
    "Menlo",
    "Monaco",
    "Consolas",
    "Liberation Mono",
    "DejaVu Sans Mono",
    "monospace",
)
QT_UI_FAMILIES = (QT_UI_FAMILY, *LANGUAGE_FONT_FAMILIES, *_SYSTEM_UI_FAMILIES)
QT_MONOSPACE_FAMILIES = (QT_MONOSPACE_FAMILY, *LANGUAGE_FONT_FAMILIES, *_SYSTEM_MONOSPACE_FAMILIES)


def _qss_stack(families: tuple[str, ...]) -> str:
    return ", ".join(
        family if family in {"sans-serif", "monospace"} else f"'{family}'"
        for family in families
    )


UI_QSS_FONT_STACK = _qss_stack(QT_UI_FAMILIES)
MONOSPACE_QSS_FONT_STACK = _qss_stack(QT_MONOSPACE_FAMILIES)
# The same stacks for a page in a browser view, which names WEB_LANGUAGE_FAMILY.
WEB_UI_FONT_STACK = _qss_stack((QT_UI_FAMILY, WEB_LANGUAGE_FAMILY, *_SYSTEM_UI_FAMILIES))
WEB_MONOSPACE_FONT_STACK = _qss_stack((QT_MONOSPACE_FAMILY, WEB_LANGUAGE_FAMILY, *_SYSTEM_MONOSPACE_FAMILIES))

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
        if sys.platform == "win32":
            # Unhinted, as the browser views draw the same fonts. Qt hints on
            # Windows, snapping each stroke to whole pixels, so Chinese, with
            # many strokes to a character, looked heavier than in the Script
            # Description. macOS never hints, and on Linux Qt and Chromium
            # both follow the system's setting, so those keep their default.
            application_font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        # A widget class with a font of its own from the platform theme keeps
        # its size and weight but takes these families too, so labels, buttons,
        # menus and tooltips draw the bundled fonts, Chinese included.
        # setFont empties Qt's table of those fonts (Studio_Theme.WIDGET_FONT_CLASSES).
        previous_font = app.font()
        class_fonts = {
            name: font for name, font in widget_class_fonts().items() if font != previous_font
        }
        app.setFont(application_font)
        for name, font in class_fonts.items():
            font = QFont(font)
            font.setFamilies(application_font.families())
            font.setHintingPreference(application_font.hintingPreference())
            app.setFont(font, name)

    return QtFontLoadStatus(
        loaded_files=tuple(loaded_files),
        failed_files=tuple(failed_files),
        loaded_families=tuple(sorted(loaded_families)),
        ui_family_available=ui_available,
        monospace_family_available=monospace_available,
    )


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
    Registered while a window shows, as a redraw in the language does, it
    then makes Qt match fonts again (_match_fonts_again).
    """
    from PySide6.QtGui import QFontDatabase

    font = LANGUAGE_FONTS.get(language)
    if font is None:
        return ()
    font_ids: list[int] = []
    registered: list[Path] = []
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
            registered.append(font_path)
    if registered and _window_shown():
        _match_fonts_again(registered[0])
    return tuple(font_ids)


def _window_shown() -> bool:
    app = QtCore.QCoreApplication.instance()
    return isinstance(app, QApplication) and any(widget.isVisible() for widget in app.topLevelWidgets())


def _match_fonts_again(font_path: Path) -> None:
    """Make Qt look fonts up again, as for windows built after it starts.

    Text a window drew before a font was added goes on finding what Qt
    found then, in new windows too, until Qt invalidates its font lookups:
    removing a font does that, adding one does not. On Windows with Noto
    Sans SC installed system-wide, a window redrawn into Simplified
    Chinese drew its bold text in the system font's thin face instead of
    the bundled Bold. Adding font_path once more and removing that copy
    leaves the language's own registration in place.
    """
    from PySide6.QtGui import QFontDatabase

    spare = QFontDatabase.addApplicationFont(str(font_path))
    if spare >= 0:
        QFontDatabase.removeApplicationFont(spare)


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


def language_web_font_css(language: str | None, url_prefix: str) -> str:
    """@font-face rules that let a browser view draw language in its bundled font, or "".

    They declare the language font, under WEB_LANGUAGE_FAMILY, with its
    regular (400) and bold (700) faces, for its script's characters
    (LanguageFont.web_range). The browser views' font stacks name that
    family after the core one (WEB_UI_FONT_STACK), as the Qt stacks name the
    language fonts, so it draws only what the core lacks. Added only for
    this language, the rules leave other languages with the system's fonts.
    url_prefix leads to the desktop font folder: "/fonts/desktop/" on the
    Viewer's web server, or a path relative to a page's base URL.
    """
    return web_font_css(LANGUAGE_FONTS.get(language), url_prefix)


# =====================================================================
# 4. Responsive Layouts (Height-for-Width & Wrapping Flows)
# =====================================================================

class ResponsiveFieldLayout(QLayout):
    """Lay out existing label/control pairs horizontally, or as readable rows.

    Stacked, every pair's label is as wide as the row's widest, so the
    controls line up; stacked_label_width widens that to a width shared with
    other rows, whose own labels differ. Such a row also takes no part in the
    window's shared label column, which stacked rows otherwise move into.
    """

    def __init__(self, parent, pairs, ratios, *, field_ratios=False,
                 trailing=False, spacing=30, column_spacing=None,
                 equal_fields=False, control_stretches=None, wrap_labels=True,
                 column_width_provider=None, stacked_label_width=None):
        super().__init__(parent)
        self.stacked_label_width = stacked_label_width
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
            width = (self._shared_label_width()
                     + self.spacing()
                     + max(self._minimum(control).width() for _, control in self.pairs))
        return QSize(width, max(self._minimum(w).height()
                               for pair in self.pairs for w in pair))

    def sizeHint(self):
        width = self._wide_width()
        return QSize(width, self.heightForWidth(width))

    def _label_width(self, label):
        return max(label.minimumWidth(), label.sizeHint().width())

    def _shared_label_width(self):
        """The width every label has when the pairs stack."""
        return max([self._label_width(label) for label, _ in self.pairs]
                   + [self.stacked_label_width or 0])

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

    def column_widths(self, width):
        """Each column's width when the row is wide and laid out in width.

        The Low Resource Mode section lines its second column up with the colour
        rows' by asking the colour row for its widths.
        """
        gap = self.spacing()
        widths = self._column_minima()
        available = width - self.column_spacing * (len(widths) - 1)
        if self.column_width_provider is not None:
            widths = self.column_width_provider(width)
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
                column = remaining * ratio // ratio_sum
                widths.append(column)
                remaining -= column
                ratio_sum -= ratio
            widths[0] += prefix
        return widths

    def _arrange(self, rect, *, apply):
        gap = self.spacing()
        row_gap = 12
        wide = rect.width() >= self._wide_width()
        if apply:
            self.parentWidget().setProperty("stacked", not wide)
        shared_label = self._shared_label_width()

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

        widths = self.column_widths(rect.width())
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


class ResponsiveColumnsLayout(ResponsiveFlowLayout):
    """Equal columns side by side while each keeps its minimum width, else one per row.

    Unlike the flow it extends, a wide row splits its width evenly, as a box
    layout with equal stretches does, so two halves stay halves. Once a column
    would drop below its minimum, every column takes a full-width row of its own,
    and the layout's minimum width is only its widest column's.

    Its parent's "stacked" property says which it is, and ``on_stack(stacked)``,
    if given, is called once the layout has switched, after the layout pass.
    """

    def __init__(self, parent=None, spacing=6, on_stack=None):
        super().__init__(parent, spacing)
        self._on_stack = on_stack
        self._stacked = None

    def setGeometry(self, rect):
        super().setGeometry(rect)
        visible = self._visible()
        column = (rect.width() - self.spacing() * max(0, len(visible) - 1)) // max(1, len(visible))
        stacked = any(self._minimum(index).width() > column for index in visible)
        if stacked != self._stacked:
            self._stacked = stacked
            parent = self.parentWidget()
            if parent is not None:
                parent.setProperty("stacked", stacked)
            if self._on_stack is not None:
                QTimer.singleShot(0, lambda: self._on_stack(stacked))

    def sizeHint(self):
        visible = self._visible()
        width = (len(visible) * max((self._minimum(i).width() for i in visible), default=0)
                 + self.spacing() * max(0, len(visible) - 1))
        return QSize(width, self.heightForWidth(width))

    def _arrange(self, rect, *, apply):
        visible = self._visible()
        if not visible:
            return 0
        count = len(visible)
        column = (rect.width() - self.spacing() * (count - 1)) // count
        if all(self._minimum(index).width() <= column for index in visible):
            height = max(self._items[index].sizeHint().height() for index in visible)
            x = rect.x()
            for position, index in enumerate(visible):
                # The last column takes what the even split leaves over.
                width = column if position < count - 1 else rect.x() + rect.width() - x
                if apply:
                    self._items[index].setGeometry(QRect(x, rect.y(), width, height))
                x += column + self.spacing()
            return height
        y = rect.y()
        for index in visible:
            height = self._items[index].sizeHint().height()
            if apply:
                self._items[index].setGeometry(QRect(rect.x(), y, rect.width(), height))
            y += height + self.spacing()
        return y - rect.y() - self.spacing()


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
        self._placeholder.installEventFilter(self)
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

    def eventFilter(self, watched, event):
        # The application stylesheet re-resolves the label's font when it
        # polishes the label, after the box's own FontChange has passed; give
        # the box's font back whenever the label's drifts from it.
        if (watched is self._placeholder and event.type() == QEvent.Type.FontChange
                and self._placeholder.font() != self.font()):
            self._placeholder.setFont(self.font())
        return super().eventFilter(watched, event)

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


class StatisticsReport(WrappedPlaceholderTextEdit):
    """The statistics report: its text, with the threshold table drawn as a histogram.

    The table's count column is cumulative: the edges at or above each
    threshold. A row's bucket is the edges from its threshold up to the next
    one's (its count less the next row's; the last row's, its count): the
    distribution of the edges' scores, binned at the table's steps.
    Behind each data row's first cell (the threshold) lies a bar as long as
    its bucket's share of the largest bucket, from the cell's left edge, in
    the theme's histogram colour. The text stays text, to select and copy. The columns after the first keep the width of their
    widest text (CAP_CHARACTERS more), so a wide report gives the room to
    the first column and its bars; a report too narrow for that lays the
    table out as Qt does.
    """

    BAR_COLUMN = 0
    COUNT_COLUMN = 1
    CAP_CHARACTERS = 4      # room after a capped column's widest text
    CELL_GAP = 12           # statistics_report_html's padding before every column but the first
    BAR_GAP = 2             # between one row's bar and the next

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document().setUndoRedoEnabled(False)
        self._counts = None
        self._fitting = False
        self.document().contentsChanged.connect(self._contents_changed)

    def _contents_changed(self):
        if not self._fitting:
            self._counts = None
            QTimer.singleShot(0, self._fit_columns)

    def _tables(self):
        return [frame for frame in self.document().rootFrame().childFrames()
                if isinstance(frame, QTextTable)]

    @staticmethod
    def _cell_text(table, row, column):
        cell = table.cellAt(row, column)
        cursor = cell.firstCursorPosition()
        cursor.setPosition(cell.lastCursorPosition().position(), QTextCursor.MoveMode.KeepAnchor)
        return cursor.selectedText().strip()

    def bar_values(self):
        """(table, row, bucket) of every data row whose count column holds a number."""
        if self._counts is None:
            self._counts = []
            for table in self._tables():
                if table.columns() <= max(self.BAR_COLUMN, self.COUNT_COLUMN):
                    continue
                counts = []
                for row in range(1, table.rows()):
                    try:
                        counts.append((row, float(
                            self._cell_text(table, row, self.COUNT_COLUMN).replace(",", "")
                        )))
                    except ValueError:
                        continue
                for index, (row, count) in enumerate(counts):
                    above = counts[index + 1][1] if index + 1 < len(counts) else 0.0
                    self._counts.append((table, row, max(0.0, count - above)))
        return self._counts

    def _fit_columns(self):
        """Cap the columns after the first, while the report has room for them."""
        font = self.document().defaultFont()
        heading_font = QFont(font)
        heading_font.setBold(True)  # the headings are semibold
        metrics, heading = QFontMetricsF(font), QFontMetricsF(heading_font)
        margin = self.document().documentMargin()
        available = self.viewport().width() - 2 * margin
        self._fitting = True
        try:
            for table in self._tables():
                widths = []
                for column in range(table.columns()):
                    texts = [self._cell_text(table, row, column) for row in range(1, table.rows())]
                    width = max([heading.horizontalAdvance(self._cell_text(table, 0, column))]
                                + [metrics.horizontalAdvance(text) for text in texts])
                    widths.append(math.ceil(width) + (self.CELL_GAP if column else 0))
                capped = [width + round(self.CAP_CHARACTERS * metrics.horizontalAdvance("0"))
                          for width in widths[1:]]
                constraints = []
                if widths and available >= widths[0] + sum(capped):
                    constraints = [QTextLength(QTextLength.Type.VariableLength, 0)] + [
                        QTextLength(QTextLength.Type.FixedLength, width) for width in capped
                    ]
                table_format = table.format()
                if table_format.columnWidthConstraints() != constraints:
                    table_format.setColumnWidthConstraints(constraints)
                    table.setFormat(table_format)
        finally:
            self._fitting = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_columns()

    def bar_rects(self):
        """Each bar, in document coordinates."""
        values = self.bar_values()
        largest = max((bucket for _, _, bucket in values), default=0)
        if largest <= 0:
            return []
        layout = self.document().documentLayout()
        rects = []
        for table, row, bucket in values:
            block = table.cellAt(row, self.BAR_COLUMN).firstCursorPosition().block()
            cell = layout.blockBoundingRect(block)
            if bucket > 0:
                rects.append(QRectF(cell.left(), cell.top() + self.BAR_GAP / 2,
                                    cell.width() * bucket / largest, cell.height() - self.BAR_GAP))
        return rects

    def paintEvent(self, event):
        rects = self.bar_rects()
        if rects:
            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.translate(-self.horizontalScrollBar().value(), -self.verticalScrollBar().value())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(TOKENS["histogram"]))
            for rect in rects:
                painter.drawRoundedRect(rect, 3, 3)
            painter.end()
        # The document draws its text over the bars.
        super().paintEvent(event)


# =====================================================================
# 5. Combo Boxes: Steady Lists, and Values Apart from Their Labels
# =====================================================================

class _ListScrollGuard(QtCore.QObject):
    """Tells a combo box's list view which kind of input drives it.

    A list view scrolls to its current row whenever that row changes (its
    autoScroll property). The mouse changes the current row too: the popup
    makes the row under the pointer the current one, and a press or a drag
    sets it as well. So resting the pointer on a partly visible last row
    scrolled it into view, which showed the next partly visible row under the
    pointer, and the list ran away under the mouse. The guard turns autoScroll
    off for every mouse event and on again for every key press, ahead of the
    popup's own filter and the view, so a key that moves the highlight still
    keeps it in view and the mouse never scrolls the list. The wheel and the
    scroll bar are not affected: they change no current row.
    """

    _MOUSE_EVENTS = frozenset({
        QEvent.Type.MouseMove,
        QEvent.Type.MouseButtonPress,
        QEvent.Type.MouseButtonRelease,
        QEvent.Type.MouseButtonDblClick,
    })

    def __init__(self, view):
        super().__init__(view)
        self._view = view
        # Mouse events reach the viewport; key presses reach the view.
        view.installEventFilter(self)
        view.viewport().installEventFilter(self)

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind in self._MOUSE_EVENTS:
            self._view.setAutoScroll(False)
        elif kind == QEvent.Type.KeyPress:
            self._view.setAutoScroll(True)
        return False


class SteadyPopupComboBox(QComboBox):
    """A combo box whose open list scrolls only by the wheel, the scroll bar or the keys.

    Qt scrolls an open list under a resting pointer: the row under it becomes
    the current row, and the list scrolls to show that row whole (see
    _ListScrollGuard). Every combo box of the programs derives from this one.
    The guard is put on the list the first time it opens: a window holds many
    combo boxes and most are never opened. The highlight still follows the
    pointer, a click selects the row it is on, and the list opens at the
    current entry.
    """

    def showPopup(self):
        view = self.view()
        if view.findChild(_ListScrollGuard) is None:
            _ListScrollGuard(view)
        # The list opens scrolled to its current entry, whichever input the
        # previous opening ended on. Its first mouse event turns scrolling by
        # the current row off, and its first key turns it on again.
        view.setAutoScroll(True)
        super().showPopup()


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


# Qt's styles leave this gap between a button's icon and its text.
BUTTON_ICON_SPACING = 4


def button_icon_room(button):
    """Return the width a button's icon takes before its text: the icon and the gap after it."""
    if button.icon().isNull():
        return 0
    return button.iconSize().width() + (BUTTON_ICON_SPACING if button.text() else 0)


def clipped_button_text(button):
    """Return how many pixels of the button's text its style has no room for.

    A style can keep a frame or stylesheet padding inside the button, so the
    room left for the text is narrower than the button. An icon takes its
    width out of that room too.
    """
    option = QStyleOptionButton()
    button.initStyleOption(option)
    room = button.style().subElementRect(
        QStyle.SubElement.SE_PushButtonContents, option, button
    ).width() - button_icon_room(button)
    text = shown_button_text(button.text())
    return max(0, button.fontMetrics().horizontalAdvance(text) - room)


def fit_buttons_to_text(*buttons, padding=BUTTON_TEXT_PADDING):
    """Give ``buttons`` one fixed width: their widest text and icon plus ``padding`` at each end.

    Should a style keep more room inside a button than the padding, the
    width grows by what it would clip, so no text is ever cut off.
    """
    width = 0
    for button in buttons:
        button.ensurePolished()  # A stylesheet may set the font the text is drawn in.
        width = max(
            width,
            text_button_width([button.text()], button.font(), padding) + button_icon_room(button),
        )
    for button in buttons:
        button.setFixedWidth(width)
    width += max(clipped_button_text(button) for button in buttons)
    for button in buttons:
        button.setFixedWidth(width)
    return width


TOGGLE_SWITCH_HEIGHT = 28


class ToggleSwitch(QPushButton):
    """A checkable switch that reads ``on_text`` or ``off_text``, "ON" and "OFF" by default.

    Studio_Theme.paint_switch draws it as a track and a knob, after the words
    both texts share ("Auto" of "Auto ON" and "Auto OFF"). The text stays the
    button's text, for screen readers and the translation checks. Its width
    fits the longer of the two texts in bold, as the pill it replaced did, so
    the layouts keep their sizes, and toggling never resizes it.
    """

    def __init__(self, on_text=None, off_text=None, parent=None):
        super().__init__(parent)
        if on_text is None:
            on_text = QtCore.QCoreApplication.translate("ToggleSwitch", "ON")
        if off_text is None:
            off_text = QtCore.QCoreApplication.translate("ToggleSwitch", "OFF")
        self._texts = (off_text, on_text)
        self.setCheckable(True)
        self.setStyleSheet(SWITCH_STYLESHEET)
        bold = QFont(self.font())
        bold.setBold(True)
        label = switch_label(self._texts)
        drawn = SWITCH_WIDTH + (
            QFontMetrics(self.font()).horizontalAdvance(label) + SWITCH_LABEL_GAP if label else 0
        )
        self.setFixedSize(max(text_button_width(self._texts, bold), drawn), TOGGLE_SWITCH_HEIGHT)
        self.toggled.connect(self._show_state)
        self._show_state(False)
        self._focus_ring = SwitchFocusRing(self)

    def state_texts(self):
        """The texts it reads off and on, whichever shows now."""
        return self._texts

    def _show_state(self, checked):
        self.setText(self._texts[bool(checked)])
        self.update()

    def paintEvent(self, event):
        paint_switch(self)


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
    previous_punctuation: object = None

    def remove(self):
        global _installed_translations
        from PySide6.QtGui import QFontDatabase

        for translator in self.translators:
            QtCore.QCoreApplication.removeTranslator(translator)
        set_translator(self.previous_message_translator)
        set_punctuation(self.previous_punctuation)
        for font_id in self.font_ids:
            QFontDatabase.removeApplicationFont(font_id)
        if _installed_translations is self:
            _installed_translations = None


# QCoreApplication does not own its translators, so this keeps them alive.
_installed_translations: InstalledTranslations | None = None


def _translate_message(template, n=-1):
    return translate(MESSAGE_CONTEXT, template, None, n)


def _load_catalog(prefix, language, directory):
    """A translator holding Qt's own <directory>/<prefix>_<language>.qm, or None."""
    translator = QtCore.QTranslator()
    if translator.load(QtCore.QLocale(language), prefix, "_", str(directory)):
        return translator
    return None


def _load_language_catalog(name, language, languages_dir):
    """A translator holding <languages_dir>/<language>/<name>_<language>.qm, or None.

    The folder and the file must be named exactly so, as on Linux, whatever
    the system (Language_Packs.find_catalog).
    """
    path = find_catalog(language, name, ".qm", languages_dir)
    translator = QtCore.QTranslator()
    if path is not None and translator.load(str(path)):
        return translator
    return None


def _language_punctuation(language, languages_dir):
    """The Punctuation of language's language.json in languages_dir, or None."""
    try:
        return read_pack(language, languages_dir).punctuation
    except ManifestError as error:
        _warn_once(f"{error} Its messages join without its punctuation.")
        return None


def install_translations(app, language, catalog_dir=LANGUAGES_DIR, extra_catalogs=()):
    """Show the windows built after this call in language.

    None or "en" keeps every text English as written and installs nothing.
    PSEUDO_LANGUAGE installs the test-only pseudo-language made from
    <catalog_dir>/emapssn.ts. Any other language loads Qt's own catalogs
    for it, where Qt has them, then its folder's catalog,
    <catalog_dir>/<language>/emapssn_<language>.qm, which wins where both
    translate a text. Without that catalog it raises LookupError and
    installs nothing. Message texts (the Viewer's console line) then come
    from the same catalogs, and messages put together from parts take the
    punctuation its language.json gives. A language whose script the core
    fonts lack also registers its bundled font (LANGUAGE_FONTS).

    extra_catalogs lists (directory, name) pairs of catalogs a window adds
    to the main one, as VR Config adds opt_vr's emapssn_vr: the
    pseudo-language is made from <directory>/<name>.ts too, and a language
    loads <directory>/<language>/<name>_<language>.qm after the main
    catalog. One that is missing leaves its own texts English.

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
    punctuation = None
    if language == PSEUDO_LANGUAGE:
        translators = [pseudo_translator(Path(catalog_dir) / f"{CATALOG_NAME}.ts")]
        translators += [
            pseudo_translator(directory / f"{name}.ts")
            for directory, name in extra_catalogs
            if (directory / f"{name}.ts").is_file()
        ]
    else:
        ours = _load_language_catalog(CATALOG_NAME, language, catalog_dir)
        if ours is None:
            raise LookupError(
                f"No {CATALOG_NAME} catalog for language {language!r} in {catalog_dir}: "
                f"it would be {Path(catalog_dir) / str(language) / f'{CATALOG_NAME}_{language}.qm'}."
            )
        punctuation = _language_punctuation(language, catalog_dir)
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
            if (translator := _load_language_catalog(name, language, directory)) is not None
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
        set_punctuation(punctuation),
    )
    return _installed_translations


def installed_language():
    """The language install_translations put in place, or None for English."""
    return None if _installed_translations is None else _installed_translations.language


def message_translator(language, catalog_dir=LANGUAGES_DIR):
    """translate(template, n) for Message texts in language, for a process with no windows.

    A tool's matplotlib chart shows Message.display() text in the language
    the windows show, but the tool has no QApplication to install Qt's
    translators in. This asks the same catalog directly: the language's
    <catalog_dir>/<language>/emapssn_<language>.qm, or for PSEUDO_LANGUAGE
    the pseudo-language made from <catalog_dir>/emapssn.ts. Pass the result
    to Localization.set_translator. None or "en" gives None, English as
    written. Without the language's catalog it raises LookupError, as
    install_translations does.
    """
    if language in (None, "", ENGLISH):
        return None
    if language == PSEUDO_LANGUAGE:
        catalog = pseudo_translator(Path(catalog_dir) / f"{CATALOG_NAME}.ts")
    else:
        catalog = _load_language_catalog(CATALOG_NAME, language, catalog_dir)
        if catalog is None:
            raise LookupError(f"No {CATALOG_NAME} catalog for language {language!r} in {catalog_dir}.")

    def translate_message(template, n=-1):
        # Qt answers "" or None for a text it lacks, which then shows in English.
        shown = catalog.translate(MESSAGE_CONTEXT, template, None, n) or template
        return english_text(shown, n)

    return translate_message


# =====================================================================
# 8. The Language Option: Setting, Dropdown & Redraw
# =====================================================================

LANGUAGE_SETTING = "LANGUAGE"
SYSTEM_LANGUAGE = "system"
ENGLISH = "en"
ENGLISH_NAME = "English"


def catalog_languages(catalog_dir=LANGUAGES_DIR):
    """The languages that have a compiled catalog in their folder, by code, sorted.

    Each language's folder in catalog_dir is <code>/, holding
    emapssn_<code>.qm (utilities/Language_Packs.py). A language.json in such
    a folder that can't be used is warned about here, once.
    """
    for problem in read_packs(catalog_dir)[1]:
        _warn_once(f"{problem} The language uses the program's defaults instead.")
    return [code for code in language_codes(catalog_dir) if code != ENGLISH]


def language_name(code, catalog_dir=LANGUAGES_DIR):
    """A language's name in that language, as the Language dropdown lists it.

    The "name" of its language.json, else Qt's name for the code. A name
    a language.json gives tells apart what Qt names alike, such as
    Simplified and Traditional Chinese.
    """
    if code == ENGLISH:
        return ENGLISH_NAME
    try:
        name = read_pack(code, catalog_dir).name
    except ManifestError:
        name = None  # catalog_languages warns about it.
    if name:
        return name
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


class LanguageSelector(SteadyPopupComboBox):
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
        system = language_name(system_language(available, ui_languages), catalog_dir)
        self.addItem(
            QtCore.QCoreApplication.translate("LanguageSelector", "System default ({language})")
            .format(language=system),
            SYSTEM_LANGUAGE,
        )
        self.addItem(language_name(ENGLISH), ENGLISH)
        for code in available:
            self.addItem(language_name(code, catalog_dir), code)
        self.setToolTip(QtCore.QCoreApplication.translate(
            "LanguageSelector",
            "The language of the windows. This window switches at once, and another open "
            "Config or Tools window when it is next clicked into; other windows use it "
            "the next time they open.",
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


class LanguageFollower(QtCore.QObject):
    """Brings a window round to the language another window saved, when it is next activated.

    Every window shares one LANGUAGE setting, but each shows the language it
    was built in, so a choice made in Tools leaves an open Config as it was.
    A window that installs this, as LanguageFollower(self), with a
    language_selector and a switch_language(language) method, looks at the
    saved setting each time it becomes the active window, as when it is
    clicked into: its dropdown shows the setting, and switch_language
    redraws it in the language the setting names, if that isn't the one it
    shows. The redrawn window looks again when it is activated in turn, and
    finds its language.

    It looks once the activation is over, after any mouse button that is
    down goes up, so the click that activated the window reaches it whole.
    It leaves the window as it is, quietly, while app_settings.json can't be
    read or holds no LANGUAGE (no window saved one), while a modal dialog or
    a popup such as an open dropdown shows, when the next activation looks
    again, and while the window shows the test-only pseudo-language. A
    redraw that fails leaves the window as it is, with a warning, and that
    setting isn't tried again.
    """

    BUTTON_WAIT_MS = 50     # between looks while a mouse button is down

    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self._failed_setting = None
        window.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.WindowActivate:
            # Not during the activation, which a redraw would replace the window in.
            QTimer.singleShot(0, self, self.follow)
        return False

    def follow(self):
        """Show the saved language now, if the window may change; returns the window that shows it."""
        window = self._window
        if not window.isActiveWindow():
            return window  # Left again before this looked; the next activation looks.
        if QApplication.activeModalWidget() is not None or QApplication.activePopupWidget() is not None:
            return window
        if QApplication.mouseButtons() != Qt.MouseButton.NoButton:
            QTimer.singleShot(self.BUTTON_WAIT_MS, self, self.follow)
            return window
        if installed_language() == PSEUDO_LANGUAGE:
            return window
        try:
            settings = read_app_settings()
        except AppSettingsError:
            return window
        if LANGUAGE_SETTING not in settings:
            return window
        setting = configured_language(settings)
        if setting == self._failed_setting:
            return window
        try:
            shown = window.switch_language(resolve_language(setting))
        except Exception as error:
            self._failed_setting = setting
            _warn_once(f"The window could not switch to the saved language {setting!r}, so it stays as it is: {error}")
            return window
        # "System default (English)" and "English" show the same language.
        shown.language_selector.show_setting(setting)
        return shown


def language_selector_row(selector, theme_switch=None, leading=None):
    """A row that ends in the Language dropdown, after a globe, and then theme_switch if given.

    leading, if given, starts the row and takes all the room the rest leaves.
    """
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    if leading is None:
        row.addStretch()
    else:
        row.addWidget(leading, 1)
    globe = IconLabel("globe")
    globe.setObjectName("languageGlobe")
    globe.setBuddy(selector)
    row.addWidget(globe)
    row.addWidget(selector)
    if theme_switch is not None:
        row.addSpacing(8)
        row.addWidget(theme_switch)
    return row


# =====================================================================
# 9. The Theme Colour Option
# =====================================================================

# Each window keeps its theme colour in app_settings.json under its own
# setting. The Viewer has none: it takes Config's when it starts, and its
# Agent and Meta Data pages take the Viewer's.
CONFIG_THEME_SETTING = "CONFIG_THEME_COLOR"
VR_CONFIG_THEME_SETTING = "VR_CONFIG_THEME_COLOR"
TOOLS_THEME_SETTING = "TOOLS_THEME_COLOR"


def theme_color_name(color):
    """A theme colour's name in the window's language."""
    names = {
        "purple": QtCore.QCoreApplication.translate("ThemeColorSwitch", "Purple"),
        "blue": QtCore.QCoreApplication.translate("ThemeColorSwitch", "Blue"),
        "teal": QtCore.QCoreApplication.translate("ThemeColorSwitch", "Teal"),
        "green": QtCore.QCoreApplication.translate("ThemeColorSwitch", "Green"),
        "orange": QtCore.QCoreApplication.translate("ThemeColorSwitch", "Orange"),
        "red": QtCore.QCoreApplication.translate("ThemeColorSwitch", "Red"),
        "pink": QtCore.QCoreApplication.translate("ThemeColorSwitch", "Pink"),
        "black": QtCore.QCoreApplication.translate("ThemeColorSwitch", "Black"),
    }
    return names.get(color, color)


def configured_theme_color(setting, settings=None):
    """The theme colour saved under setting, else DEFAULT_THEME_COLOR.

    settings stands in for app_settings.json. A file that can't be read, or
    a colour the program doesn't have, counts as no setting, with a warning.
    """
    if settings is None:
        try:
            settings = read_app_settings()
        except AppSettingsError as error:
            _warn_once(f"{error} The windows take the default theme colour.")
            settings = {}
    value = settings.get(setting)
    if value is None or value in THEME_COLORS:
        return value or DEFAULT_THEME_COLOR
    _warn_once(f"There is no theme colour named {value!r}; the window takes the default.")
    return DEFAULT_THEME_COLOR


def theme_swatch(color):
    """The colour of a theme colour's circle: its primary button's."""
    return QColor(theme_tokens(color)["primary"])


def _paint_swatch(painter, rect, color):
    painter.setPen(QPen(QColor(0, 0, 0, 28), 1.0))
    painter.setBrush(theme_swatch(color))
    painter.drawEllipse(rect)


def _paint_ring(painter, rect, color, gap=3.0):
    pen = QPen(QColor(color))
    pen.setWidthF(2.0)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(rect.adjusted(-gap - 0.5, -gap - 0.5, gap + 0.5, gap + 0.5))


class ThemeColorPalette(QWidget):
    """Every theme colour as a circle in a row, on a card that pops up.

    The colour shown now is ringed in itself, and the one under the mouse, or
    picked with the arrow keys, in grey, with its name below the row. A click,
    Enter or Space emits chosen with that colour and closes the palette; Esc,
    or a click outside it, closes it without a choice.
    """

    chosen = QtCore.Signal(str)

    SWATCH = 20     # a circle's diameter
    GAP = 8         # between circles
    PAD = 10        # inside the card
    NAME_GAP = 6    # between the circles and the name
    RADIUS = 10     # the card's corners
    # Where the window system composes translucent windows, the card has round
    # corners and paints its own shadow in this margin; elsewhere it fills the window.
    SHADOW = 8

    def __init__(self, current, parent=None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.NoDropShadowWindowHint)
        self._translucent = (sys.platform in ("win32", "darwin")
                             or QApplication.platformName() in ("wayland", "offscreen"))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, self._translucent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(QtCore.QCoreApplication.translate("ThemeColorSwitch", "Theme Color"))
        self.colors = list(THEME_COLORS)
        self.current = current
        self.hovered = self.colors.index(current) if current in self.colors else 0
        metrics = self.fontMetrics()
        row = len(self.colors) * self.SWATCH + (len(self.colors) - 1) * self.GAP
        names = max(metrics.horizontalAdvance(theme_color_name(color)) for color in self.colors)
        margin = self.SHADOW if self._translucent else 0
        self._card_size = QSize(2 * self.PAD + max(row, names),
                                2 * self.PAD + self.SWATCH + self.NAME_GAP + metrics.height())
        self.setFixedSize(self._card_size.width() + 2 * margin, self._card_size.height() + 2 * margin)

    def card_rect(self):
        """Where the card is, inside the shadow's margin when there is one."""
        margin = self.SHADOW if self._translucent else 0
        # The shadow falls a little below the card, so the card sits a little higher.
        top = margin - 2 if self._translucent else 0
        return QRectF(margin, top, self._card_size.width(), self._card_size.height())

    def swatch_rect(self, index):
        card = self.card_rect()
        row = len(self.colors) * self.SWATCH + (len(self.colors) - 1) * self.GAP
        left = card.x() + (card.width() - row) / 2
        return QRectF(left + index * (self.SWATCH + self.GAP), card.y() + self.PAD,
                      self.SWATCH, self.SWATCH)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        card = self.card_rect()
        radius = self.RADIUS if self._translucent else 0
        if self._translucent:
            painter.setPen(Qt.PenStyle.NoPen)
            for grow, alpha in ((7.0, 4), (5.0, 6), (3.0, 9), (1.5, 12)):
                painter.setBrush(QColor(9, 9, 11, alpha))
                painter.drawRoundedRect(card.adjusted(-grow, -grow + 2, grow, grow + 2),
                                        radius + grow, radius + grow)
        painter.setPen(QPen(QColor(TOKENS["border"]), 1.0))
        painter.setBrush(QColor(TOKENS["surface"]))
        painter.drawRoundedRect(card.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        for index, color in enumerate(self.colors):
            rect = self.swatch_rect(index)
            if color == self.current:
                _paint_ring(painter, rect, theme_swatch(color))
            elif index == self.hovered:
                _paint_ring(painter, rect, TOKENS["field_border_hover"])
            _paint_swatch(painter, rect, color)
        painter.setPen(QColor(TOKENS["text_muted"]))
        name = QRectF(card.x(), card.y() + self.PAD + self.SWATCH + self.NAME_GAP,
                      card.width(), self.fontMetrics().height())
        painter.drawText(name, Qt.AlignmentFlag.AlignCenter, theme_color_name(self.colors[self.hovered]))
        painter.end()

    def index_at(self, position):
        """The colour whose circle (with its ring) holds position, or None."""
        for index in range(len(self.colors)):
            if self.swatch_rect(index).adjusted(-4, -4, 4, 4).contains(position):
                return index
        return None

    def mouseMoveEvent(self, event):
        index = self.index_at(event.position())
        if index is not None and index != self.hovered:
            self.hovered = index
            self.update()

    def mouseReleaseEvent(self, event):
        index = self.index_at(event.position())
        if index is not None:
            self._choose(index)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            self.hovered = (self.hovered + (1 if key == Qt.Key.Key_Right else -1)) % len(self.colors)
            self.update()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._choose(self.hovered)
        else:
            super().keyPressEvent(event)

    def _choose(self, index):
        self.close()
        self.chosen.emit(self.colors[index])


class ThemeColorButton(QPushButton):
    """A window's theme colour switch: a circle in the colour; a click opens the palette.

    setting names the window's setting (CONFIG_THEME_SETTING and so on); the
    circle starts at the colour saved there, or at color. Picking a colour in
    the palette emits color_chosen with it, for choose_theme_color to save.
    """

    color_chosen = QtCore.Signal(str)
    SIZE = 28       # as tall as a field
    INSET = 5       # from the button's edge to the circle

    def __init__(self, setting, parent=None, *, color=None):
        super().__init__(parent)
        self.setObjectName("themeColorButton")
        self.theme_setting = setting
        self._color = configured_theme_color(setting) if color is None else color
        self.palette_popup = None
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # It paints itself; the stylesheet only clears the button's frame.
        self.setStyleSheet("QPushButton { background: transparent; border: none; padding: 0px; }")
        text = QtCore.QCoreApplication.translate("ThemeColorSwitch", "Theme color: {color}").format(
            color=theme_color_name(self._color)
        )
        self.setToolTip(text)
        self.setAccessibleName(text)
        self.clicked.connect(self.open_palette)

    def color(self):
        """The colour the circle shows."""
        return self._color

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self.isEnabled():
            painter.setOpacity(0.45)
        circle = QRectF(self.rect()).adjusted(self.INSET, self.INSET, -self.INSET, -self.INSET)
        if self.hasFocus():
            _paint_ring(painter, circle, TOKENS["accent"], gap=2.0)
        elif self.underMouse() and self.isEnabled():
            _paint_ring(painter, circle, TOKENS["field_border_hover"], gap=2.0)
        _paint_swatch(painter, circle, self._color)
        painter.end()

    def open_palette(self):
        """Show the palette above the circle, its right edge at the circle's; returns it."""
        popup = ThemeColorPalette(self._color, self)
        popup.chosen.connect(self.color_chosen)
        card = popup.card_rect()
        corner = self.mapToGlobal(QtCore.QPoint(self.width(), 0))
        x = corner.x() - round(card.right())
        y = corner.y() - 4 - round(card.bottom())
        screen = self.screen().availableGeometry()
        if y + card.top() < screen.top():
            # No room above, as at the top of a screen: below the circle instead.
            y = corner.y() + self.height() + 4 - round(card.top())
        popup.move(max(screen.left(), min(x, screen.right() - popup.width())), y)
        popup.show()
        popup.setFocus()
        self.palette_popup = popup
        return popup


def choose_theme_color(window, color):
    """Save color as window's theme colour, then redraw window in it.

    window has a theme_switch (a ThemeColorButton) and a
    switch_theme_color(color) method. If the colour can't be saved, a message
    says why and nothing changes. Returns the window that shows the colour.
    """
    try:
        save_app_setting(window.theme_switch.theme_setting, color)
    except (AppSettingsError, OSError) as error:
        QMessageBox.critical(
            window,
            QtCore.QCoreApplication.translate("ThemeColorSwitch", "Theme Color Not Saved"),
            QtCore.QCoreApplication.translate(
                "ThemeColorSwitch", "The theme color could not be saved, so it stays as it was.\n\n{error}"
            ).format(error=error),
        )
        return window
    return window.switch_theme_color(color)


def redraw_in_theme_color(window, color, build):
    """Replace window with a copy in the theme colour named color, as redraw_in_language does.

    The application takes the colour first, and gets the previous one back
    if build() fails. The language stays as installed.
    """
    app = QApplication.instance()
    previous = theme_color()
    installed = _installed_translations
    apply_studio_theme(app, color)
    try:
        if installed is None:
            return redraw_in_language(window, None, build)
        return redraw_in_language(window, installed.language, build,
                                  installed.catalog_dir, installed.extra_catalogs)
    except BaseException:
        apply_studio_theme(app, previous)
        raise


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
    without a window. If the language can't be installed or build() fails,
    the window and its language stay. Returns the replacement.
    """
    app = QApplication.instance()
    previous = _installed_translations
    view = capture_view(window)
    try:
        # A language whose catalog is gone raises after removing the one installed.
        install_translations(app, language, catalog_dir, extra_catalogs)
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


# =====================================================================
# 10. The Colour Picker
# =====================================================================

# The picker's preset colours, in two rows: black, greys and white, then the
# hue circle in Tailwind's 500 shades, with the Viewer's default colours
# (grey text, #ff0000, #ffaa00 and #4488ff nodes) in place of their neighbours.
COLOR_PRESETS = (
    "#000000", "#808080", "#d4d4d8", "#ffffff", "#ff0000", "#f97316", "#ffaa00", "#eab308", "#84cc16",
    "#22c55e", "#14b8a6", "#06b6d4", "#3b82f6", "#4488ff", "#6366f1", "#8b5cf6", "#d946ef", "#ec4899",
)


def color_from_text(text):
    """The colour a field's text names (#rrggbb, rrggbb, #rgb or a name such as grey), or an invalid QColor."""
    text = text.strip()
    if re.fullmatch(r"[0-9a-fA-F]{3}|[0-9a-fA-F]{6}", text):
        text = "#" + text
    return QColor.fromString(text) if text else QColor()


class ColorPicker(QWidget):
    """A colour picker on a card that pops up, as the theme palette does.

    A square in the colour's hue, saturation across and brightness down; a
    hue strip; the colour beside its hex code, which takes typing; and preset
    colours, the one shown ringed. Every change emits color_changed at once.
    Enter, or a click outside the card, closes it; Esc closes it and emits
    cancelled, for the colour it opened with to come back.
    """

    color_changed = QtCore.Signal(QColor)
    cancelled = QtCore.Signal()

    PAD = 12            # inside the card
    GAP = 12            # between its parts
    AREA_WIDTH = 248    # the square, and every part's width
    AREA_HEIGHT = 156
    HUE_HEIGHT = 12
    SWATCH = 20         # a preset's diameter
    SWATCH_GAP = 6      # between the presets' rows
    PER_ROW = 9
    FIELD_WIDTH = 112   # the hex code's field
    RADIUS = 10         # the card's corners
    SHADOW = 8          # as ThemeColorPalette's
    # The hue strip stops short of 360 degrees, which is red again, at its left.
    MAX_HUE = 359.99 / 360

    def __init__(self, color, parent=None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.NoDropShadowWindowHint)
        self._translucent = (sys.platform in ("win32", "darwin")
                             or QApplication.platformName() in ("wayland", "offscreen"))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, self._translucent)
        self.setMouseTracking(True)
        self.setAccessibleName(QtCore.QCoreApplication.translate("ColorPicker", "Select Color"))
        self.hovered = None     # the preset under the mouse
        self._dragging = None   # "area" or "hue", while the mouse drags in it
        self._hue = self._saturation = self._value = 0.0
        self._color = QColor()
        self.hex_field = QLineEdit(self)
        self.hex_field.setAccessibleName(QtCore.QCoreApplication.translate("ColorPicker", "Color code"))
        self.hex_field.textEdited.connect(self._hex_edited)
        self.hex_field.editingFinished.connect(lambda: self.hex_field.setText(self._color.name()))
        self.hex_field.returnPressed.connect(self.close)
        self._field_height = self.hex_field.sizeHint().height()
        rows = math.ceil(len(COLOR_PRESETS) / self.PER_ROW)
        self._card_size = QSize(
            self.AREA_WIDTH + 2 * self.PAD,
            2 * self.PAD + self.AREA_HEIGHT + self.HUE_HEIGHT + self._field_height + 3 * self.GAP
            + rows * self.SWATCH + (rows - 1) * self.SWATCH_GAP,
        )
        margin = self.SHADOW if self._translucent else 0
        self.setFixedSize(self._card_size.width() + 2 * margin, self._card_size.height() + 2 * margin)
        field = self.color_swatch_rect()
        self.hex_field.setGeometry(round(field.right() + 8), round(field.y()), self.FIELD_WIDTH, self._field_height)
        color = QColor(color)
        self._set_color(color if color.isValid() else QColor("#ffffff"), emit=False)

    def color(self):
        """The colour the picker shows."""
        return QColor(self._color)

    # --- Where its parts are ---

    def card_rect(self):
        """Where the card is, inside the shadow's margin when there is one."""
        margin = self.SHADOW if self._translucent else 0
        top = margin - 2 if self._translucent else 0
        return QRectF(margin, top, self._card_size.width(), self._card_size.height())

    def area_rect(self):
        card = self.card_rect()
        return QRectF(card.x() + self.PAD, card.y() + self.PAD, self.AREA_WIDTH, self.AREA_HEIGHT)

    def hue_rect(self):
        area = self.area_rect()
        return QRectF(area.x(), area.bottom() + self.GAP, self.AREA_WIDTH, self.HUE_HEIGHT)

    def color_swatch_rect(self):
        """The colour shown, a square as tall as the hex code's field beside it."""
        hue = self.hue_rect()
        return QRectF(hue.x(), hue.bottom() + self.GAP, self._field_height, self._field_height)

    def preset_rect(self, index):
        row, column = divmod(index, self.PER_ROW)
        step = (self.AREA_WIDTH - self.SWATCH) / (self.PER_ROW - 1)
        top = self.color_swatch_rect().bottom() + self.GAP
        return QRectF(self.area_rect().x() + column * step, top + row * (self.SWATCH + self.SWATCH_GAP),
                      self.SWATCH, self.SWATCH)

    def preset_at(self, position):
        """The preset whose circle (with its ring) holds position, or None."""
        for index in range(len(COLOR_PRESETS)):
            if self.preset_rect(index).adjusted(-3, -3, 3, 3).contains(position):
                return index
        return None

    # --- The colour ---

    def _set_color(self, color, *, emit=True, keep_field=False):
        """Show color as it is; a grey keeps the hue the strip shows."""
        hue = color.hsvHueF()
        if hue >= 0:
            self._hue = min(hue, self.MAX_HUE)
        self._saturation = max(0.0, color.hsvSaturationF())
        self._value = color.valueF()
        self._show(QColor(color.name()), emit, keep_field)

    def _set_hsv(self):
        self._show(QColor(QColor.fromHsvF(self._hue, self._saturation, self._value).name()), True, False)

    def _show(self, color, emit, keep_field):
        changed = color.name() != self._color.name()
        self._color = color
        if not keep_field:
            self.hex_field.setText(color.name())
        self.update()
        if emit and changed:
            self.color_changed.emit(QColor(color))

    def _hex_edited(self, text):
        color = color_from_text(text)
        if color.isValid():
            self._set_color(color, keep_field=True)

    # --- Painting ---

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        card = self.card_rect()
        radius = self.RADIUS if self._translucent else 0
        if self._translucent:
            painter.setPen(Qt.PenStyle.NoPen)
            for grow, alpha in ((7.0, 4), (5.0, 6), (3.0, 9), (1.5, 12)):
                painter.setBrush(QColor(9, 9, 11, alpha))
                painter.drawRoundedRect(card.adjusted(-grow, -grow + 2, grow, grow + 2),
                                        radius + grow, radius + grow)
        painter.setPen(QPen(QColor(TOKENS["border"]), 1.0))
        painter.setBrush(QColor(TOKENS["surface"]))
        painter.drawRoundedRect(card.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)

        # Saturation from left to right, brightness from top to bottom, in the hue.
        area = self.area_rect()
        clip = QPainterPath()
        clip.addRoundedRect(area, 8, 8)
        painter.save()
        painter.setClipPath(clip)
        across = QLinearGradient(area.topLeft(), area.topRight())
        across.setColorAt(0, QColor("#ffffff"))
        across.setColorAt(1, QColor.fromHsvF(self._hue, 1, 1))
        painter.fillRect(area, across)
        down = QLinearGradient(area.topLeft(), area.bottomLeft())
        down.setColorAt(0, QColor(0, 0, 0, 0))
        down.setColorAt(1, QColor(0, 0, 0, 255))
        painter.fillRect(area, down)
        painter.restore()
        self._paint_handle(painter, QtCore.QPointF(area.x() + self._saturation * area.width(),
                                                   area.y() + (1 - self._value) * area.height()), self._color)

        bar = self.hue_rect()
        hues = QLinearGradient(bar.topLeft(), bar.topRight())
        for step in range(7):
            hues.setColorAt(step / 6, QColor.fromHsvF((step / 6) % 1.0, 1, 1))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(hues)
        painter.drawRoundedRect(bar, bar.height() / 2, bar.height() / 2)
        self._paint_handle(painter, QtCore.QPointF(bar.x() + self._hue * bar.width(), bar.center().y()),
                           QColor.fromHsvF(self._hue, 1, 1))

        painter.setPen(QPen(QColor(0, 0, 0, 40), 1.0))
        painter.setBrush(self._color)
        painter.drawRoundedRect(self.color_swatch_rect().adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)

        for index, preset in enumerate(COLOR_PRESETS):
            rect = self.preset_rect(index)
            if preset == self._color.name():
                _paint_ring(painter, rect, TOKENS["accent"])
            elif index == self.hovered:
                _paint_ring(painter, rect, TOKENS["field_border_hover"])
            painter.setPen(QPen(QColor(0, 0, 0, 40), 1.0))
            painter.setBrush(QColor(preset))
            painter.drawEllipse(rect)
        painter.end()

    @staticmethod
    def _paint_handle(painter, center, fill):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 50))
        painter.drawEllipse(center + QtCore.QPointF(0, 1), 9, 9)
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(center, 8, 8)
        painter.setBrush(fill)
        painter.drawEllipse(center, 5.5, 5.5)

    # --- The mouse and keys ---

    def mousePressEvent(self, event):
        position = event.position()
        if not self.card_rect().contains(position):
            # Outside the card, as on its shadow, any button closes it; QWidget
            # closes a popup on a press outside the window.
            if self.rect().contains(position.toPoint()):
                self.close()
            else:
                super().mousePressEvent(event)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if self.area_rect().contains(position):
            self._dragging = "area"
        elif self.hue_rect().adjusted(0, -4, 0, 4).contains(position):
            self._dragging = "hue"
        else:
            index = self.preset_at(position)
            if index is not None:
                self._set_color(QColor(COLOR_PRESETS[index]))
            return
        self._drag_to(position)

    def mouseMoveEvent(self, event):
        position = event.position()
        if self._dragging:
            self._drag_to(position)
            return
        index = self.preset_at(position)
        if index != self.hovered:
            self.hovered = index
            self.update()
        over_part = (index is not None or self.area_rect().contains(position)
                     or self.hue_rect().adjusted(0, -4, 0, 4).contains(position))
        self.setCursor(Qt.CursorShape.PointingHandCursor if over_part else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._drag_to(event.position())
            self._dragging = None

    def _drag_to(self, position):
        if self._dragging == "area":
            area = self.area_rect()
            self._saturation = min(1.0, max(0.0, (position.x() - area.x()) / area.width()))
            self._value = 1.0 - min(1.0, max(0.0, (position.y() - area.y()) / area.height()))
        else:
            bar = self.hue_rect()
            self._hue = min(self.MAX_HUE, max(0.0, (position.x() - bar.x()) / bar.width()))
        self._set_hsv()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.close()
        else:
            super().keyPressEvent(event)


def open_color_picker(button, field):
    """Open a ColorPicker under button for the colour field (a QLineEdit) holds; return it.

    The card's right edge lines up with the button's, below it, or above it
    where the screen has no room below. Each colour picked goes into field
    at once, as #rrggbb; Esc puts back the text field held. A field with no
    colour opens at white.
    """
    original = field.text()
    picker = ColorPicker(color_from_text(original), button)
    picker.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    picker.color_changed.connect(lambda color: field.setText(color.name()))
    picker.cancelled.connect(lambda: field.setText(original))
    card = picker.card_rect()
    top_left = button.mapToGlobal(QtCore.QPoint(0, 0))
    x = top_left.x() + button.width() - round(card.right())
    y = top_left.y() + button.height() + 4 - round(card.top())
    screen = button.screen().availableGeometry()
    if y + card.bottom() > screen.bottom():
        y = top_left.y() - 4 - round(card.bottom())
    picker.move(max(screen.left(), min(x, screen.right() - picker.width())), y)
    picker.show()
    picker.hex_field.setFocus()
    picker.hex_field.selectAll()
    return picker


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
    "instance_handover_message",
    "SingleInstanceController",
    "show_window_in_front",
    "open_in_file_manager",
    "QT_UI_FAMILY",
    "QT_MONOSPACE_FAMILY",
    "VISPY_UI_FACE",
    "VISPY_MONOSPACE_FACE",
    "VISPY_FALLBACK_FACE",
    "VISPY_REFERENCE_DPI",
    "QT_UI_FAMILIES",
    "QT_MONOSPACE_FAMILIES",
    "UI_QSS_FONT_STACK",
    "MONOSPACE_QSS_FONT_STACK",
    "WEB_UI_FONT_STACK",
    "WEB_MONOSPACE_FONT_STACK",
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
    "LANGUAGE_FONT_FAMILIES",
    "LANGUAGE_FONT_FILES",
    "WEB_LANGUAGE_FAMILY",
    "vispy_points_for_logical_pixels",
    "vispy_points_at_reference_dpi",
    "QtFontLoadStatus",
    "VispyFontLoadStatus",
    "configure_qt_application_fonts",
    "apply_studio_theme",
    "qt_monospace_font",
    "register_vispy_application_fonts",
    "vispy_language_face",
    "language_web_font_css",
    "ResponsiveFieldLayout",
    "ResponsiveFlowLayout",
    "ResponsiveColumnsLayout",
    "ResponsiveSelectorLayout",
    "WrappedPlaceholderTextEdit",
    "StatisticsReport",
    "SteadyPopupComboBox",
    "add_combo_options",
    "combo_value",
    "select_combo_value",
    "NAME_ITEM_ROLE",
    "mark_name_item",
    "BUTTON_TEXT_PADDING",
    "shown_button_text",
    "text_button_width",
    "clipped_button_text",
    "BUTTON_ICON_SPACING",
    "button_icon_room",
    "fit_buttons_to_text",
    "TOGGLE_SWITCH_HEIGHT",
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
    "ENGLISH_NAME",
    "catalog_languages",
    "language_name",
    "system_language",
    "configured_language",
    "resolve_language",
    "LanguageSelector",
    "CONFIG_THEME_SETTING",
    "VR_CONFIG_THEME_SETTING",
    "TOOLS_THEME_SETTING",
    "ThemeColorButton",
    "ThemeColorPalette",
    "theme_swatch",
    "choose_theme_color",
    "configured_theme_color",
    "redraw_in_theme_color",
    "theme_color_name",
    "choose_language",
    "LanguageFollower",
    "language_selector_row",
    "capture_view",
    "restore_view",
    "restore_view_positions",
    "redraw_in_language",
    "COLOR_PRESETS",
    "ColorPicker",
    "color_from_text",
    "open_color_picker",
]
