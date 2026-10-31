# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The BLASTP Directory row of the Tools window (desktop.Blast_Directory_Control).

A blank field means "search automatically" and shows what that search finds as
its placeholder; the Download and Browse icon buttons come after it; the
download flow (confirmation, folder, worker thread, progress, result) runs with
install_blast replaced. No test reaches the network, and every dialog is
replaced, since an unmocked modal box would hang an offscreen run.
"""

import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTextBrowser,
)

import EMAPSSN_Tools  # noqa: E402
from desktop import Blast_Directory_Control as control_module  # noqa: E402
from desktop.Blast_Directory_Control import BlastDirectoryControl, display_path  # noqa: E402
from tests.tools_gui_fixtures import isolated_tools_project  # noqa: E402
from utilities import BLAST_Download, BLAST_Executables  # noqa: E402

WINDOWS = "windows-x86_64"
AUTOMATIC = "C:\\Program Files\\NCBI\\blast-2.17.0+\\bin (standard install folder)"
SUBSTITUTION_TOOL = "Align_Substitution_Matrix.py"


def fake_install(folder_name="blast-2.17.0+", version="blastp: 2.17.0+"):
    """An install_blast that reports each stage and returns where it 'installed'."""
    calls = []

    def install(destination, progress=None, cancel=None, **options):
        calls.append(destination)
        progress(10_000_000, 140_000_000, "download")
        progress(1_000, 3_000, "extract")
        progress(0, 0, "check")
        return BLAST_Download.BlastInstall(os.path.join(destination, folder_name), version)

    install.calls = calls
    return install


class BlockedInstall:
    """An install_blast that reports one step, then waits for its cancel flag (or release()) before it ends."""

    def __init__(self, error=None):
        self.error = error
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.destinations = []
        self._release = threading.Event()

    def __call__(self, destination, progress=None, cancel=None, **options):
        self.destinations.append(destination)
        progress(5_000_000, 140_000_000, "download")
        self.started.set()
        while not cancel.is_set():
            if self._release.wait(0.005):
                if self.error:
                    raise self.error
                return BLAST_Download.BlastInstall(os.path.join(destination, "blast-2.17.0+"), "blastp: 2.17.0+")
        self.cancelled.set()
        raise BLAST_Download.BlastDownloadCancelled("The BLAST+ download was cancelled.")

    def release(self):
        self._release.set()


class ControlTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.project = os.path.join(self.root.name, "project")
        os.makedirs(self.project)
        self.outside = os.path.join(self.root.name, "elsewhere")
        os.makedirs(self.outside)

    def flush(self):
        for _ in range(4):
            self.app.processEvents()

    def wait_until(self, condition, what="the condition", seconds=10):
        deadline = time.monotonic() + seconds
        while not condition():
            self.app.processEvents()
            if time.monotonic() > deadline:
                self.fail(f"timed out waiting for {what}")
            time.sleep(0.005)

    def patch(self, target, attribute, **options):
        patcher = mock.patch.object(target, attribute, **options)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def supported_host(self):
        self.patch(BLAST_Download, "unsupported_reason", return_value=None)
        self.patch(BLAST_Download, "platform_key", return_value=WINDOWS)

    def dialogs(self, *, answer="Yes", folder=None):
        """Replace the confirmation, the folder picker and the result boxes; returns them by name."""
        yes_no = QMessageBox.StandardButton
        mocks = {
            "question": self.patch(
                QMessageBox, "question", return_value=yes_no.Yes if answer == "Yes" else yes_no.No
            ),
            "folder": self.patch(
                QFileDialog, "getExistingDirectory", return_value=self.project if folder is None else folder
            ),
            "information": self.patch(QMessageBox, "information"),
            "critical": self.patch(QMessageBox, "critical"),
        }
        return mocks

    def control(self, value=""):
        self.supported_host()
        self.patch(BLAST_Executables, "describe_automatic", return_value=AUTOMATIC)
        control = BlastDirectoryControl(value, self.project)
        self.addCleanup(control.deleteLater)
        self.addCleanup(control.stop_download)
        return control


class PathTextTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)

    def test_a_folder_inside_the_project_is_relative(self):
        inside = os.path.join(self.root.name, "tools", "blast-2.17.0+")
        self.assertEqual(display_path(inside, self.root.name), os.path.join("tools", "blast-2.17.0+"))

    def test_a_folder_outside_the_project_is_absolute(self):
        outside = os.path.join(os.path.dirname(self.root.name), "somewhere", "blast-2.17.0+")
        self.assertEqual(display_path(outside, self.root.name), os.path.normpath(outside))

    def test_a_sibling_whose_name_starts_like_the_project_is_not_inside_it(self):
        sibling = self.root.name + "-other"
        self.assertEqual(display_path(os.path.join(sibling, "b"), self.root.name), os.path.join(sibling, "b"))

    def test_the_project_itself_and_another_drive_stay_absolute(self):
        self.assertEqual(display_path(self.root.name, self.root.name), os.path.normpath(self.root.name))
        with mock.patch.object(control_module.os.path, "commonpath", side_effect=ValueError):
            folder = os.path.join(self.root.name, "b")
            self.assertEqual(display_path(folder, self.root.name), os.path.normpath(folder))


class FieldTests(ControlTestCase):
    def test_a_blank_field_shows_what_the_automatic_search_finds(self):
        control = self.control()
        self.assertEqual(control.line_edit.text(), "")
        self.assertEqual(control.line_edit.placeholderText(), f"Automatic: {AUTOMATIC}")

    def test_it_says_so_when_the_search_finds_nothing(self):
        self.supported_host()
        self.patch(BLAST_Executables, "describe_automatic", return_value=BLAST_Executables.NOT_FOUND)
        control = BlastDirectoryControl("", self.project)
        self.addCleanup(control.deleteLater)
        self.assertEqual(
            control.line_edit.placeholderText(), "No BLAST+ found: download it or browse to its folder"
        )

    def test_a_value_it_is_given_is_kept(self):
        control = self.control("C:/saved/blast")
        self.assertEqual(control.line_edit.text(), "C:/saved/blast")

    def test_download_comes_first_then_browse_each_with_an_icon_and_a_tooltip(self):
        control = self.control()
        row = control.layout()
        widgets = [row.itemAt(index).widget() for index in range(row.count())]
        self.assertEqual(widgets, [control.line_edit, control.download_button, control.browse_button])
        version = BLAST_Download.read_pin()["version"]
        self.assertEqual(control.download_button.toolTip(), f"Download BLAST+ {version}…")
        self.assertEqual(control.browse_button.toolTip(), "Browse…")
        for button in (control.download_button, control.browse_button):
            self.assertFalse(button.icon().isNull())
            self.assertEqual(button.text(), "")
            self.assertEqual(button.accessibleName(), button.toolTip())

    def test_the_two_buttons_have_different_icons(self):
        control = self.control()
        size = control.download_button.iconSize()
        download = control.download_button.icon().pixmap(size).toImage()
        browse = control.browse_button.icon().pixmap(size).toImage()
        self.assertNotEqual(download, browse)

    def test_browse_starts_at_the_field_and_a_relative_one_is_read_against_the_project(self):
        control = self.control(os.path.join("tools", "blast"))
        picked = self.patch(QFileDialog, "getExistingDirectory", return_value=self.outside)
        control.browse_button.click()
        self.assertEqual(picked.call_args.args[2], os.path.join(self.project, "tools", "blast"))
        self.assertEqual(control.line_edit.text(), os.path.normpath(self.outside))
        picked.return_value = ""
        control.browse_button.click()
        self.assertEqual(control.line_edit.text(), os.path.normpath(self.outside))

    def test_an_unsupported_host_disables_download_and_says_why(self):
        self.patch(BLAST_Download, "unsupported_reason", return_value="No build for this computer.")
        self.patch(BLAST_Executables, "describe_automatic", return_value=AUTOMATIC)
        dialogs = self.dialogs()
        control = BlastDirectoryControl("", self.project)
        self.addCleanup(control.deleteLater)
        self.assertFalse(control.download_button.isEnabled())
        self.assertEqual(control.download_button.toolTip(), "No build for this computer.")
        self.assertTrue(control.browse_button.isEnabled())
        control.start_download()
        dialogs["question"].assert_not_called()


class DownloadFlowTests(ControlTestCase):
    def run_download(self, control, install):
        self.patch(BLAST_Download, "install_blast", new=install)
        control.download_button.click()
        self.wait_until(lambda: not control.is_downloading(), "the download to end")
        self.flush()

    def test_the_confirmation_names_the_version_the_sizes_the_licence_and_that_the_window_stays_usable(self):
        control = self.control()
        dialogs = self.dialogs(answer="No")
        control.download_button.click()
        dialogs["question"].assert_called_once()
        text = dialogs["question"].call_args.args[2]
        pin = BLAST_Download.read_pin()
        entry = pin["platforms"][WINDOWS]
        self.assertIn(f"BLAST+ {pin['version']}", text)
        self.assertIn("NCBI (ftp.ncbi.nlm.nih.gov)", text)
        self.assertIn(f"about {round(entry['size'] / 1e6)} MB", text)
        self.assertIn(f"about {round(entry['installed_size'] / 1e6)} MB", text)
        self.assertIn("public domain", text)
        self.assertIn("stays usable", text)

    def test_declining_the_confirmation_does_nothing(self):
        control = self.control("kept")
        dialogs = self.dialogs(answer="No")
        install = self.patch(BLAST_Download, "install_blast")
        control.download_button.click()
        dialogs["folder"].assert_not_called()
        install.assert_not_called()
        self.assertFalse(control.is_downloading())
        self.assertTrue(control.download_button.isEnabled())
        self.assertEqual(control.line_edit.text(), "kept")

    def test_cancelling_the_folder_picker_does_nothing(self):
        control = self.control("kept")
        dialogs = self.dialogs(folder="")
        install = self.patch(BLAST_Download, "install_blast")
        control.download_button.click()
        self.assertEqual(dialogs["folder"].call_args.args[2], self.project)  # starts at the project root
        install.assert_not_called()
        self.assertFalse(control.is_downloading())
        self.assertEqual(control.line_edit.text(), "kept")
        dialogs["information"].assert_not_called()
        dialogs["critical"].assert_not_called()

    def test_a_success_inside_the_project_sets_a_relative_folder_and_names_the_version(self):
        control = self.control("old")
        dialogs = self.dialogs(folder=os.path.join(self.project, "tools"))
        install = fake_install()
        self.run_download(control, install)
        self.assertEqual(install.calls, [os.path.join(self.project, "tools")])
        self.assertEqual(control.line_edit.text(), os.path.join("tools", "blast-2.17.0+"))
        dialogs["critical"].assert_not_called()
        dialogs["information"].assert_called_once()
        message = dialogs["information"].call_args.args[2]
        self.assertIn("blastp: 2.17.0+", message)
        self.assertIn(os.path.join(self.project, "tools", "blast-2.17.0+"), message)
        self.assertTrue(control.download_button.isEnabled())

    def test_a_success_outside_the_project_sets_an_absolute_folder(self):
        control = self.control()
        self.dialogs(folder=self.outside)
        self.run_download(control, fake_install())
        self.assertEqual(control.line_edit.text(), os.path.join(os.path.normpath(self.outside), "blast-2.17.0+"))

    def test_a_failure_shows_its_message_and_leaves_the_field(self):
        control = self.control("C:/old/blast")
        dialogs = self.dialogs()
        message = "BLAST+ was downloaded and checked, but blastp cannot be started. Install the Visual C++ runtime."
        install = BlockedInstall(BLAST_Download.BlastDownloadError(message))
        self.patch(BLAST_Download, "install_blast", new=install)
        control.download_button.click()
        self.wait_until(install.started.is_set, "the install to start")
        self.assertTrue(control.is_downloading())
        self.assertFalse(control.download_button.isEnabled())
        install.release()
        self.wait_until(lambda: not control.is_downloading(), "the install to fail")
        dialogs["critical"].assert_called_once()
        self.assertEqual(dialogs["critical"].call_args.args[2], message)
        dialogs["information"].assert_not_called()
        self.assertEqual(control.line_edit.text(), "C:/old/blast")
        self.assertTrue(control.download_button.isEnabled())

    def test_an_unexpected_error_is_shown_too(self):
        control = self.control("kept")
        dialogs = self.dialogs()
        install = mock.Mock(side_effect=RuntimeError("disk exploded"))
        self.run_download(control, install)
        self.assertEqual(dialogs["critical"].call_args.args[2], "disk exploded")
        self.assertEqual(control.line_edit.text(), "kept")

    def test_the_progress_dialog_shows_the_stage_and_the_window_is_not_blocked(self):
        control = self.control()
        self.dialogs()
        install = BlockedInstall()
        self.patch(BLAST_Download, "install_blast", new=install)
        control.download_button.click()
        self.wait_until(install.started.is_set, "the install to start")
        dialog = control._progress
        self.assertIsInstance(dialog, QProgressDialog)
        self.assertFalse(dialog.isModal())
        self.wait_until(lambda: dialog.labelText().startswith("Downloading BLAST+"), "the progress text")
        self.assertIn("5 of 140 MB", dialog.labelText())
        self.assertEqual(dialog.maximum(), 140_000_000 // 1024)
        # The field can be typed in while it runs.
        control.line_edit.setText("typed meanwhile")
        self.assertEqual(control.line_edit.text(), "typed meanwhile")
        control._show_progress(1_000, 3_000, "extract")
        self.assertIn("Unpacking BLAST+", dialog.labelText())
        control._show_progress(0, 0, "check")
        self.assertIn("Checking that BLAST+", dialog.labelText())
        self.assertEqual((dialog.minimum(), dialog.maximum()), (0, 0))
        install.release()
        self.wait_until(lambda: not control.is_downloading(), "the install to end")

    def test_the_cancel_button_sets_the_event_and_the_field_stays(self):
        control = self.control("kept")
        dialogs = self.dialogs()
        install = BlockedInstall()
        self.patch(BLAST_Download, "install_blast", new=install)
        control.download_button.click()
        self.wait_until(install.started.is_set, "the install to start")
        dialog = control._progress
        cancel = next(button for button in dialog.findChildren(QPushButton) if button.text() == "Cancel")
        self.assertFalse(control._worker.cancel.is_set())
        cancel.click()
        self.assertTrue(control._worker.cancel.is_set())
        self.wait_until(lambda: not control.is_downloading(), "the install to stop")
        self.assertTrue(install.cancelled.is_set())
        self.assertEqual(control.line_edit.text(), "kept")
        dialogs["critical"].assert_not_called()
        dialogs["information"].assert_not_called()
        self.assertTrue(control.download_button.isEnabled())
        self.assertIsNone(control._progress)

    def test_a_progress_report_after_cancel_does_not_show_the_dialog_again(self):
        control = self.control()
        self.dialogs()
        install = BlockedInstall()
        self.patch(BLAST_Download, "install_blast", new=install)
        control.download_button.click()
        self.wait_until(install.started.is_set, "the install to start")
        dialog = control._progress
        control.cancel_download()
        dialog.hide()
        control._show_progress(6_000_000, 140_000_000, "download")
        self.assertFalse(dialog.isVisible())
        self.wait_until(lambda: not control.is_downloading(), "the install to stop")

    def test_a_second_click_while_it_runs_starts_nothing(self):
        control = self.control()
        dialogs = self.dialogs()
        install = BlockedInstall()
        self.patch(BLAST_Download, "install_blast", new=install)
        control.download_button.click()
        self.wait_until(install.started.is_set, "the install to start")
        control.start_download()
        self.assertEqual(dialogs["question"].call_count, 1)
        self.assertEqual(len(install.destinations), 1)
        install.release()
        self.wait_until(lambda: not control.is_downloading(), "the install to end")

    def test_stop_download_cancels_and_joins_without_reporting(self):
        control = self.control("kept")
        dialogs = self.dialogs()
        install = BlockedInstall()
        self.patch(BLAST_Download, "install_blast", new=install)
        control.download_button.click()
        self.wait_until(install.started.is_set, "the install to start")
        worker = control._worker
        control.stop_download()
        self.assertTrue(install.cancelled.is_set())
        self.assertTrue(worker.isFinished())
        self.assertFalse(control.is_downloading())
        self.flush()
        dialogs["critical"].assert_not_called()
        dialogs["information"].assert_not_called()
        self.assertEqual(control.line_edit.text(), "kept")


class ToolsWindowRowTests(ControlTestCase):
    """The row as the Tools window builds it."""

    def window(self, sections=None):
        self.supported_host()
        self.patch(BLAST_Executables, "describe_automatic", return_value=AUTOMATIC)
        root = isolated_tools_project(self, sections)
        self.project = str(root)
        with mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser):
            window = EMAPSSN_Tools.ToolsGUI()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        path = next(path for path in window.script_data if path.endswith(SUBSTITUTION_TOOL))
        return window, path, window.script_data[path]["inputs"]["BLASTP_DIR"]["widget"]

    def test_the_row_is_blank_with_the_placeholder_and_a_blank_value_is_what_is_saved(self):
        window, path, control = self.window()
        self.assertIsInstance(control, BlastDirectoryControl)
        self.assertEqual(control.line_edit.text(), "")
        self.assertEqual(control.line_edit.placeholderText(), f"Automatic: {AUTOMATIC}")
        self.assertEqual(window._collect_tool_settings(path)["BLASTP_DIR"], "")

    def test_a_blast_plus_on_this_computer_is_never_filled_in_or_saved(self):
        # The old row filled in what the search found and saved it with Save & Run.
        with tempfile.TemporaryDirectory() as found:
            for name in BLAST_Executables.PROGRAMS:
                Path(found, BLAST_Executables.program_file(name)).write_bytes(b"")
            with mock.patch.object(BLAST_Executables.shutil, "which",
                                   side_effect=lambda name: str(Path(found, BLAST_Executables.program_file(name)))):
                window, path, control = self.window()
                self.assertEqual(control.line_edit.text(), "")
                self.assertEqual(window._collect_tool_settings(path)["BLASTP_DIR"], "")

    def test_a_value_saved_before_stays_as_it_is(self):
        saved = os.path.normpath("C:/my/blast/bin")
        window, path, control = self.window({SUBSTITUTION_TOOL: {"BLASTP_DIR": saved}})
        self.assertEqual(control.line_edit.text(), saved)
        self.assertEqual(window._collect_tool_settings(path)["BLASTP_DIR"], saved)

    def test_a_saved_null_is_blank_and_not_the_word_none(self):
        _window, _path, control = self.window({SUBSTITUTION_TOOL: {"BLASTP_DIR": None}})
        self.assertEqual(control.line_edit.text(), "")

    def test_only_this_row_has_the_two_icon_buttons_and_the_other_browse_buttons_stay(self):
        window, _path, control = self.window()
        browse = [button for button in window.findChildren(QPushButton) if button.text() == "Browse..."]
        # One Browse button per directory of the Directories tab, and none beside the BLASTP field.
        self.assertEqual(len(browse), len(window.dir_inputs))
        self.assertFalse(any(control.isAncestorOf(button) for button in browse))
        self.assertEqual(len(window.findChildren(BlastDirectoryControl)), 1)

    def test_a_download_fills_the_row_and_it_is_collected_like_a_typed_value(self):
        window, path, control = self.window()
        self.dialogs(folder=os.path.join(self.project, "tools"))
        self.patch(BLAST_Download, "install_blast", new=fake_install())
        control.download_button.click()
        self.wait_until(lambda: not control.is_downloading(), "the download to end")
        expected = os.path.join("tools", "blast-2.17.0+")
        self.assertEqual(window._collect_tool_settings(path)["BLASTP_DIR"], expected)

    def test_closing_the_window_during_a_download_cancels_it_and_waits_for_the_thread(self):
        window, _path, control = self.window()
        dialogs = self.dialogs()
        install = BlockedInstall()
        self.patch(BLAST_Download, "install_blast", new=install)
        window.show()
        self.flush()
        control.download_button.click()
        self.wait_until(install.started.is_set, "the install to start")
        worker = control._worker
        window.close()
        self.assertTrue(install.cancelled.is_set())
        self.assertTrue(worker.isFinished())
        self.assertFalse(control.is_downloading())
        self.assertFalse(window.isVisible())
        self.flush()
        dialogs["critical"].assert_not_called()
        dialogs["information"].assert_not_called()

    def test_closing_the_window_with_no_download_just_closes(self):
        window, _path, control = self.window()
        window.show()
        self.flush()
        window.close()
        self.assertFalse(window.isVisible())
        self.assertFalse(control.is_downloading())


if __name__ == "__main__":
    unittest.main()
