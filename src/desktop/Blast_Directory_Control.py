# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The BLASTP Directory row of the Tools window: a field and its Download and Browse buttons.

A blank field means "search for BLAST+ automatically" (utilities/BLAST_Executables.py),
and the field's placeholder says what that search would use right now. Nothing is
ever put in the field for the user: only a download, the Browse button or typing do.

The Download button installs the pinned NCBI BLAST+ (utilities/BLAST_Download.py)
in a folder the user picks, after a confirmation that names the version, the
sizes and the licence. The install runs on a worker thread, so the window stays
usable; a progress dialog shows the stage and has a Cancel button. A success
fills the field with the new folder, a failure or a cancel leaves it as it was,
and closing the window cancels a download and waits for its thread
(stop_download).
"""

import os
import threading
from urllib.parse import urlsplit

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QWidget,
)

from desktop.Desktop_App import translate
from desktop.Studio_Theme import icon_button
from utilities import BLAST_Download, BLAST_Executables

ICON_BUTTON_WIDTH = 30
MEGABYTE = 1e6  # the unit the installer itself reports sizes in


def automatic_placeholder():
    """What the empty field shows: where the automatic search looks right now, or that it finds nothing."""
    place = BLAST_Executables.describe_automatic()
    if place == BLAST_Executables.NOT_FOUND:
        return translate("Tools", "No BLAST+ found: download it or browse to its folder")
    return translate("Tools", "Automatic: {place}").format(place=place)


def display_path(folder, project_root):
    """The text for the field: ``folder`` relative to ``project_root`` when it is inside, else absolute."""
    folder = os.path.abspath(folder)
    root = os.path.abspath(project_root)
    try:
        inside = os.path.commonpath([os.path.normcase(root), os.path.normcase(folder)]) == os.path.normcase(root)
    except ValueError:  # another drive
        inside = False
    if inside and os.path.normcase(folder) != os.path.normcase(root):
        return os.path.relpath(folder, root)
    return os.path.normpath(folder)


class BlastInstallWorker(QThread):
    """Runs install_blast off the window's thread.

    progress carries (done, total, stage) as install_blast reports them;
    completed carries the BlastInstall, or None and the error. Setting ``cancel``
    stops the install at its next chunk, and it ends with BlastDownloadCancelled.
    """

    progress = Signal(object, object, str)
    completed = Signal(object, object)

    def __init__(self, destination):
        super().__init__()
        self.destination = destination
        self.cancel = threading.Event()

    def run(self):
        try:
            installed = BLAST_Download.install_blast(
                self.destination, progress=self.progress.emit, cancel=self.cancel
            )
        except Exception as error:  # a BlastDownloadError, or whatever else went wrong
            self.completed.emit(None, error)
        else:
            self.completed.emit(installed, None)


class BlastDirectoryControl(QWidget):
    """The field and its two buttons: Download BLAST+ first, then Browse."""

    def __init__(self, value="", project_root="", parent=None):
        super().__init__(parent)
        self.project_root = project_root
        self._worker = None
        self._progress = None
        self._pin = None
        self._reason = BLAST_Download.unsupported_reason()

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.line_edit = QLineEdit("" if value is None else str(value))
        self.line_edit.setPlaceholderText(automatic_placeholder())

        self.download_button = icon_button("download", self._download_tooltip())
        self.download_button.setObjectName("blastDownloadButton")
        self.download_button.setFixedWidth(ICON_BUTTON_WIDTH)
        self.download_button.clicked.connect(self.start_download)
        self.browse_button = icon_button("folder-open", translate("Tools", "Browse…"))
        self.browse_button.setObjectName("blastBrowseButton")
        self.browse_button.setFixedWidth(ICON_BUTTON_WIDTH)
        self.browse_button.clicked.connect(self.browse)
        self._update_download_button()

        row.addWidget(self.line_edit)
        row.addWidget(self.download_button)
        row.addWidget(self.browse_button)

    # -- the buttons ---------------------------------------------------------

    def _version(self):
        try:
            if self._pin is None:
                self._pin = BLAST_Download.read_pin()
            return self._pin["version"]
        except BLAST_Download.BlastDownloadError:
            return ""

    def _download_tooltip(self):
        if self._reason:
            return self._reason
        version = self._version()
        if version:
            return translate("Tools", "Download BLAST+ {version}…").format(version=version)
        return translate("Tools", "Download BLAST+…")

    def _update_download_button(self):
        self.download_button.setEnabled(self._reason is None and self._worker is None)

    def _field_folder(self):
        """The folder the field names, a relative one read against the project root; "" when it is blank."""
        text = self.line_edit.text().strip()
        if text and not os.path.isabs(text):
            text = os.path.join(self.project_root, text)
        return text

    def browse(self, checked=False):
        folder = QFileDialog.getExistingDirectory(
            self.window(), translate("Tools", "Select Directory"), self._field_folder()
        )
        if folder:
            self.line_edit.setText(os.path.normpath(folder))

    # -- the download --------------------------------------------------------

    def is_downloading(self):
        return self._worker is not None

    def _confirm(self):
        """Ask whether to download, saying what it fetches, what it costs and its licence; True on a yes."""
        title = translate("Tools", "Download BLAST+")
        try:
            pin = BLAST_Download.read_pin()
        except BLAST_Download.BlastDownloadError as error:
            QMessageBox.critical(self.window(), title, str(error))
            return False
        entry = pin["platforms"].get(BLAST_Download.platform_key())
        if entry is None:
            QMessageBox.critical(
                self.window(), title, translate("Tools", "EMAP-SSN has no BLAST+ download for this computer.")
            )
            return False
        answer = QMessageBox.question(
            self.window(),
            title,
            translate(
                "Tools",
                "Download NCBI BLAST+ {version}?\n\n"
                "It comes from NCBI ({host}). The download is about {download} MB, and the two "
                "programs EMAP-SSN uses (blastp and makeblastdb) take about {installed} MB once "
                "installed. The download is checked against the SHA-256 checksums that EMAP-SSN "
                "records before anything is installed.\n\n"
                "BLAST+ is a United States government work and is in the public domain.\n\n"
                "The window stays usable while it downloads, and you can cancel at any time. "
                "You choose the folder to install it in next.",
            ).format(
                version=pin["version"],
                host=urlsplit(entry["url"]).hostname,
                download=round(entry["size"] / MEGABYTE),
                installed=round(entry["installed_size"] / MEGABYTE),
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return answer == QMessageBox.StandardButton.Yes

    def start_download(self, checked=False):
        if self._worker is not None or self._reason:
            return
        if not self._confirm():
            return
        folder = QFileDialog.getExistingDirectory(
            self.window(), translate("Tools", "Select the Folder to Install BLAST+ In"), self.project_root
        )
        if not folder:
            return
        self._worker = worker = BlastInstallWorker(os.path.normpath(folder))
        worker.progress.connect(self._show_progress)
        worker.completed.connect(self._completed)
        self._update_download_button()
        self._open_progress()
        worker.start()

    def _open_progress(self):
        # Not modal: the window stays usable, as the confirmation says.
        dialog = QProgressDialog(
            translate("Tools", "Starting the download…"),
            translate("Tools", "Cancel"),
            0,
            0,
            self.window(),
        )
        dialog.setWindowTitle(translate("Tools", "Download BLAST+"))
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        dialog.canceled.connect(self.cancel_download)
        self._progress = dialog
        dialog.show()

    def _show_progress(self, done, total, stage):
        dialog = self._progress
        if dialog is None or self._worker is None or self._worker.cancel.is_set():
            return  # a progress report after Cancel would show the dialog again
        version = self._version()
        if stage == "download":
            text = translate("Tools", "Downloading BLAST+ {version}: {done} of {total} MB").format(
                version=version, done=round(done / MEGABYTE), total=round(total / MEGABYTE)
            )
        elif stage == "extract":
            text = translate("Tools", "Unpacking BLAST+ {version}…").format(version=version)
        else:
            text = translate("Tools", "Checking that BLAST+ {version} starts…").format(version=version)
        dialog.setLabelText(text)
        if total:
            kilobytes = 1024  # a progress dialog counts in an int
            dialog.setRange(0, max(1, total // kilobytes))
            dialog.setValue(min(done // kilobytes, dialog.maximum()))
        else:
            dialog.setRange(0, 0)  # no size to count against: a busy bar

    def cancel_download(self):
        """Ask the install to stop; its thread ends with a cancel a moment later."""
        if self._worker is not None:
            self._worker.cancel.set()

    def _close_progress(self):
        dialog, self._progress = self._progress, None
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()

    def _finish_worker(self):
        worker, self._worker = self._worker, None
        self._close_progress()
        if worker is not None:
            worker.wait()  # it has emitted completed and is about to return
            worker.deleteLater()
        self._update_download_button()

    def _completed(self, installed, error):
        self._finish_worker()
        if error is not None:
            if not isinstance(error, BLAST_Download.BlastDownloadCancelled):
                QMessageBox.critical(self.window(), translate("Tools", "Download BLAST+"), str(error))
            return
        self.line_edit.setText(display_path(installed.folder, self.project_root))
        QMessageBox.information(
            self.window(),
            translate("Tools", "BLAST+ Installed"),
            translate(
                "Tools",
                "BLAST+ is installed ({version}) in:\n{folder}\n\n"
                "The BLASTP Directory field now names that folder. It is saved with the tool's "
                "settings when you Save & Run.",
            ).format(version=installed.version, folder=installed.folder),
        )

    def stop_download(self):
        """Cancel a download in progress and wait for its thread; nothing is reported afterwards.

        The install looks at its cancel flag for every chunk it reads, so the
        wait is short unless the connection has stalled, which ends in its own
        30 second timeout.
        """
        worker = self._worker
        if worker is None:
            return
        worker.cancel.set()
        worker.completed.disconnect(self._completed)
        worker.progress.disconnect(self._show_progress)
        self._finish_worker()
