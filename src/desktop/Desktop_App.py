# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Consolidated Desktop Qt application presentation, windowing, fonts, and layouts."""

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
from PySide6.QtWidgets import QLayout, QPushButton

if TYPE_CHECKING:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

# =====================================================================
# 1. Application Identity & Linux Desktop Configuration
# =====================================================================

PRODUCT_NAME = "EMAP-SSN"
APPLICATION_VERSION = "0.1.0"
PRODUCT_LONG_NAME = (
    f"{PRODUCT_NAME}: Embedding- and Multiple-Alignment-integrated Protein "
    "Sequence Similarity Network Platform"
)
CONFIG_DISPLAY_NAME = f"{PRODUCT_NAME} Configuration"
VIEWER_DISPLAY_NAME = f"{PRODUCT_NAME} Viewer"
TOOLS_DISPLAY_NAME = f"{PRODUCT_NAME} Tools"

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
VISPY_UI_FACE = "NotoSans"
VISPY_MONOSPACE_FACE = "NotoSansMono"
VISPY_FALLBACK_FACE = "OpenSans"
VISPY_REFERENCE_DPI = 96.0

QT_UI_FAMILIES = (
    QT_UI_FAMILY,
    "Segoe UI",
    ".AppleSystemUIFont",
    "Helvetica Neue",
    "Cantarell",
    "DejaVu Sans",
    "sans-serif",
)
QT_MONOSPACE_FAMILIES = (
    QT_MONOSPACE_FAMILY,
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
    return tuple(relative_path for relative_path, _ in entries)


FONT_MANIFEST_ENTRIES = _manifest_entries(FONT_MANIFEST)
FONT_FILES = tuple(relative_path for relative_path, _ in FONT_MANIFEST_ENTRIES)

UI_REGULAR_FILE = "noto/NotoSans/NotoSans-Regular.ttf"
UI_BOLD_FILE = "noto/NotoSans/NotoSans-Bold.ttf"
MONOSPACE_REGULAR_FILE = "noto/NotoSansMono/NotoSansMono-Regular.ttf"
MONOSPACE_BOLD_FILE = "noto/NotoSansMono/NotoSansMono-Bold.ttf"


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
    "VISPY_UI_FACE",
    "VISPY_MONOSPACE_FACE",
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
    "vispy_points_for_logical_pixels",
    "vispy_points_at_reference_dpi",
    "QtFontLoadStatus",
    "VispyFontLoadStatus",
    "configure_qt_application_fonts",
    "force_light_palette",
    "qt_monospace_font",
    "register_vispy_application_fonts",
    "ResponsiveFieldLayout",
    "ResponsiveFlowLayout",
    "ResponsiveSelectorLayout",
]
