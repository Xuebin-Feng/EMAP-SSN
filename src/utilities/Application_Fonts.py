# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""Cross-platform application font, palette, and VisPy registration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING
import warnings
import weakref

if TYPE_CHECKING:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication


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
