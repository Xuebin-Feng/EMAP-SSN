# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Session card's Background jobs section (desktop.Viewer_Jobs_Card, desktop.Viewer_Jobs).

The section lists the scheduler's lanes, the running job first and then those waiting,
with each job's state, progress and log, and buttons that move a waiting job, remove it,
stop the running one (after a confirmation) and dismiss a failure. These tests drive it
with a stand-in source (the rows and what the buttons call), then with the real
scheduler behind ViewerJobs, and place it in the card between the files and the controls.
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from desktop.Viewer_Jobs import JobLane, JobRow, ViewerJobs, job_lanes, job_lines  # noqa: E402
from desktop.Viewer_Jobs_Card import ElidedLabel, JobsSection, clock  # noqa: E402
from desktop.Viewer_Widgets import ControlsButton  # noqa: E402
from tests.command_fixtures import end_scheduler  # noqa: E402
from tests.test_viewer_widgets import RecordingCanvas, flush, shipped_actions  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402
from utilities.Localization import Message  # noqa: E402

LONG_NAME = "Label_Output_with_a_very_long_name_that_cannot_possibly_fit_in_the_card_20261024_101500_attempt_7.xlsx"


def running(job_id, description, started_at=100.0, **extra):
    return JobRow(job_id, "running", description, f"/out/{job_id}", 1, started_at, **extra)


def queued(job_id, description, position, up=True, down=True, **extra):
    return JobRow(job_id, "queued", description, f"/out/{job_id}", position, can_move_up=up, can_move_down=down, **extra)


class FakeJobs:
    """The rows the section lists, and a record of what its buttons asked for."""

    def __init__(self, lanes=()):
        self.current = list(lanes)
        self.calls = []

    def lanes(self):
        return list(self.current)

    def move(self, job_id, offset):
        self.calls.append(("move", job_id, offset))

    def remove(self, job_id):
        self.calls.append(("remove", job_id))

    def dismiss(self, job_id):
        self.calls.append(("dismiss", job_id))

    def open_log(self, job_id):
        self.calls.append(("log", job_id))


def files_lane(*rows, title="Files"):
    return JobLane("files", title, tuple(rows))


def dispose(widget):
    """Close a top-level test widget for good, so that the next test's window is the active one."""
    widget.hide()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


class SectionCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def make(self, lanes=(), confirm=None, now=lambda: 105.0, width=520):
        self.source = FakeJobs(lanes)
        self.asked = []

        def answer(job_id, description):
            self.asked.append((job_id, description))
            return confirm if confirm is not None else True

        self.section = JobsSection(self.source, confirm=answer, now=now)
        self.addCleanup(dispose, self.section)
        self.section.resize(width, 300)
        self.section.show()
        self.section.activateWindow()
        QTest.qWaitForWindowActive(self.section)
        flush(self.app)
        return self.section

    def texts(self):
        return [(widget.lead.full_text(), widget.state.full_text()) for widget in self.section.rows.values()]


class SectionContentTests(SectionCase):
    def test_with_no_job_one_muted_line_says_so_under_the_heading(self):
        section = self.make()
        self.assertEqual(section.heading.text(), "Background jobs")
        self.assertEqual(section.empty.text(), "No background jobs.")
        self.assertFalse(section.empty.isHidden())
        self.assertEqual((section.rows, section.lane_labels), ({}, {}))
        self.assertFalse(section.timer.isActive())

    def test_one_running_and_two_waiting_jobs_are_listed_in_order_under_their_lane(self):
        section = self.make([files_lane(
            running(1, "print -> map.png"), queued(2, "label -> report.xlsx", 2, up=False),
            queued(3, "logo -> logo.svg", 3, down=False))])
        self.assertTrue(section.empty.isHidden())
        self.assertEqual(section.lane_labels["files"].text(), "Files")
        self.assertEqual(self.texts(), [
            ("#1  print -> map.png", "Running · 00:05"),
            ("#2  label -> report.xlsx", "Queued · position 2"),
            ("#3  logo -> logo.svg", "Queued · position 3"),
        ])
        self.assertTrue(section.timer.isActive(), "the running job's clock ticks")

    def test_each_lane_is_a_group_with_its_title_from_the_job_not_from_the_card(self):
        section = self.make([
            files_lane(running(1, "a")),
            JobLane("folding", Message("Folding"), (running(2, "b"), queued(3, "c", 2, up=False, down=False))),
            JobLane("plugin lane", "Anything a plugin calls it", (queued(4, "d", 1, up=False, down=False),)),
        ])
        self.assertEqual([label.text() for label in section.lane_labels.values()],
                         ["Files", "Folding", "Anything a plugin calls it"])
        self.assertEqual(list(section.rows), [1, 2, 3, 4])

    def test_a_long_name_is_cut_short_with_an_ellipsis_and_hovering_shows_all_of_it(self):
        section = self.make([files_lane(running(1, f"label -> {LONG_NAME}"), queued(2, "label -> short.xlsx", 2))], width=420)
        long_row, short_row = section.rows[1], section.rows[2]
        self.assertTrue(long_row.lead.is_elided())
        self.assertIn(LONG_NAME, long_row.lead.toolTip())
        self.assertIn("/out/1", long_row.lead.toolTip(), "the whole output path too")
        self.assertFalse(short_row.lead.is_elided())
        # The label asks for no width of its own: a long name never widens the card.
        self.assertLess(long_row.lead.sizeHint().width(), 100)
        self.assertLessEqual(long_row.lead.width(), section.width())

    def test_the_elided_text_ends_in_an_ellipsis_when_painted(self):
        section = self.make([files_lane(running(1, f"label -> {LONG_NAME}"))], width=300)
        label = section.rows[1].lead
        shown = label.fontMetrics().elidedText(label.full_text(), Qt.TextElideMode.ElideRight, label.width())
        self.assertTrue(shown.endswith("…"))
        self.assertLess(len(shown), len(label.full_text()))

    def test_a_screen_reader_gets_the_description_and_the_state_of_a_row_whole_and_current(self):
        from PySide6.QtGui import QAccessible

        section = self.make([files_lane(running(1, f"label -> {LONG_NAME}"), queued(2, "logo -> logo.svg", 2))], width=420)
        long_row, short_row = section.rows[1], section.rows[2]
        self.assertTrue(long_row.lead.is_elided())

        def read(label):
            """What the platform's accessibility interface reports as the label's name."""
            return QAccessible.queryAccessibleInterface(label).text(QAccessible.Text.Name)

        # The label paints its own text, so QLabel's text is empty: the name is the whole text, cut or not.
        self.assertEqual(long_row.lead.text(), "")
        for label in (long_row.lead, long_row.state, short_row.lead, short_row.state):
            with self.subTest(label=label.full_text()):
                self.assertTrue(label.full_text())
                self.assertEqual(label.accessibleName(), label.full_text())
                self.assertEqual(read(label), label.full_text())
        self.assertEqual(read(long_row.lead), f"#1  label -> {LONG_NAME}")
        self.assertEqual(read(short_row.state), "Queued · position 2")
        # It follows the text: the running job's clock moves on, and the name with it.
        self.assertEqual(read(long_row.state), "Running · 00:05")
        long_row.refresh_state(107.0)
        self.assertEqual(long_row.state.full_text(), "Running · 00:07")
        self.assertEqual(read(long_row.state), "Running · 00:07")
        label = ElidedLabel()
        self.addCleanup(label.deleteLater)
        self.assertEqual(label.accessibleName(), "")
        label.set_full_text("one")
        label.set_full_text("two", "a tip")
        self.assertEqual(label.accessibleName(), "two")

    def test_a_job_shows_its_own_progress_after_its_state(self):
        section = self.make([files_lane(running(1, "esmfold", progress="Folding 2 of 3: node_b"),
                                        queued(2, "esmfold", 2, progress=Message("Waiting")))])
        self.assertEqual(section.rows[1].state.full_text(), "Running · 00:05 · Folding 2 of 3: node_b")
        self.assertEqual(section.rows[2].state.full_text(), "Queued · position 2 · Waiting")

    def test_a_job_being_stopped_says_so_and_cannot_be_removed_again(self):
        section = self.make([files_lane(JobRow(1, "stopping", "print -> a.png", "/out/1", 1, 100.0))])
        row = section.rows[1]
        self.assertEqual(row.state.full_text(), "Stopping…")
        self.assertFalse(row.remove.isEnabled())
        self.assertTrue(section.timer.isActive())

    def test_a_job_that_failed_shows_why_in_the_danger_colour_with_dismiss_and_open_log(self):
        failed = JobRow(7, "failed", "esmfold -> node", "", error=Message("Folding failed: {why}", why="out of memory"),
                        log_path="/logs/run.log")
        section = self.make([files_lane(failed)])
        row = section.rows[7]
        self.assertEqual(row.state.full_text(), "Failed: Folding failed: out of memory")
        self.assertTrue(row.state.property("failed"))
        self.assertIsNotNone(row.log_button)
        self.assertEqual(row.log_button.text(), "Open log")
        self.assertEqual(row.remove.accessibleName(), "Dismiss #7")
        self.assertFalse(section.timer.isActive(), "a failure has no clock")

    def test_the_open_log_link_shows_for_any_job_that_names_a_log(self):
        section = self.make([files_lane(running(1, "a", log_path="/logs/a.log"), queued(2, "b", 2),
                                        queued(3, "c", 3, log_path="/logs/c.log"))])
        self.assertEqual([row.log_button is not None for row in section.rows.values()], [True, False, True])


class SectionButtonTests(SectionCase):
    def lanes(self):
        return [files_lane(running(1, "print -> map.png"),
                           queued(2, "label -> a.xlsx", 2, up=False), queued(3, "logo -> b.svg", 3),
                           queued(4, "logo -> c.svg", 4, down=False))]

    def test_a_running_job_has_only_a_remove_button_and_a_waiting_one_has_up_down_and_remove(self):
        section = self.make(self.lanes())
        running_row, first, middle, last = (section.rows[number] for number in (1, 2, 3, 4))
        self.assertEqual((running_row.up, running_row.down), (None, None))
        self.assertEqual(running_row.remove.toolTip(), "Stop and remove")
        for row in (first, middle, last):
            self.assertEqual(row.remove.toolTip(), "Remove from the queue")
        self.assertEqual([(row.up.isEnabled(), row.down.isEnabled()) for row in (first, middle, last)],
                         [(False, True), (True, True), (True, False)], "the first cannot go up and the last cannot go down")

    def test_every_button_has_a_tooltip_an_accessible_name_with_its_job_and_is_reached_by_tab(self):
        section = self.make(self.lanes())
        buttons = [button for button in section.findChildren(QPushButton) if button.property("role")]
        self.assertEqual(len(buttons), 1 + 3 * 3)
        for button in buttons:
            job = button.property("job")
            self.assertTrue(button.toolTip(), button.property("role"))
            self.assertIn(f"#{job}", button.accessibleName())
            self.assertEqual(button.focusPolicy(), Qt.FocusPolicy.TabFocus, "Tab reaches it, a click does not give it the keyboard")
        names = {section.rows[3].up.accessibleName(), section.rows[3].down.accessibleName(), section.rows[3].remove.accessibleName()}
        self.assertEqual(names, {"Move #3 up", "Move #3 down", "Remove #3 from the queue"})

    def test_move_up_and_move_down_move_the_waiting_job_one_place(self):
        section = self.make(self.lanes())
        section.rows[3].up.click()
        section.rows[3].down.click()
        self.assertEqual(self.source.calls, [("move", 3, -1), ("move", 3, 1)])

    def test_a_disabled_move_button_does_nothing(self):
        section = self.make(self.lanes())
        QTest.mouseClick(section.rows[2].up, Qt.MouseButton.LeftButton)
        QTest.mouseClick(section.rows[4].down, Qt.MouseButton.LeftButton)
        self.assertEqual(self.source.calls, [])

    def test_a_waiting_job_is_removed_at_once_without_asking(self):
        section = self.make(self.lanes())
        section.rows[3].remove.click()
        self.assertEqual(self.source.calls, [("remove", 3)])
        self.assertEqual(self.asked, [])

    def test_a_running_job_is_removed_only_after_the_user_confirms(self):
        section = self.make(self.lanes(), confirm=False)
        section.rows[1].remove.click()
        self.assertEqual(self.asked, [(1, "print -> map.png")])
        self.assertEqual(self.source.calls, [], "declined: nothing is stopped")
        section.confirm = lambda job, description: self.asked.append((job, description)) or True
        section.rows[1].remove.click()
        self.assertEqual(self.source.calls, [("remove", 1)])

    def test_dismiss_and_open_log_ask_the_source_for_that_job(self):
        failed = JobRow(9, "failed", "esmfold", "", error="boom", log_path="/logs/x.log")
        section = self.make([files_lane(failed)])
        section.rows[9].log_button.click()
        section.rows[9].remove.click()
        self.assertEqual(self.source.calls, [("log", 9), ("dismiss", 9)])

    def test_the_confirmation_says_the_job_stops_at_its_next_step_and_its_unfinished_file_goes(self):
        from desktop import Viewer_Jobs_Card

        with mock.patch.object(Viewer_Jobs_Card, "QMessageBox") as box_class:
            box = box_class.return_value
            box.clickedButton.return_value = box.addButton.return_value
            self.assertTrue(Viewer_Jobs_Card.default_confirm(None, 5, "print -> map.png"))
        box.setText.assert_called_once_with("Stop job #5 while it is running?")
        informative = box.setInformativeText.call_args.args[0]
        self.assertIn("print -> map.png", informative)
        self.assertIn("next step", informative)
        self.assertIn("deleted", informative)


class SectionLiveTests(SectionCase):
    def test_a_change_to_the_rows_rebuilds_them_and_nothing_else_does(self):
        clock_now = [105.0]
        section = self.make([files_lane(running(1, "a"), queued(2, "b", 2, down=False))], now=lambda: clock_now[0])
        before = section.rows[1]
        self.assertFalse(section.refresh(), "nothing changed")
        self.assertIs(section.rows[1], before, "the rows were not rebuilt")
        clock_now[0] = 190.0
        self.assertFalse(section.refresh())
        self.assertEqual(before.state.full_text(), "Running · 01:30", "only the clock moved")
        self.source.current = [files_lane(running(1, "a"), queued(2, "b", 2, down=False), queued(3, "c", 3, up=True, down=False))]
        self.assertTrue(section.refresh())
        self.assertEqual(list(section.rows), [1, 2, 3])
        self.source.current = []
        self.assertTrue(section.refresh())
        self.assertTrue(section.empty.isVisible())
        self.assertFalse(section.timer.isActive(), "no clock to tick")

    def test_rebuilding_the_rows_opens_no_window_of_its_own(self):
        # A row taken out of its parent becomes a window, which the layout's pending show would open.
        section = self.make([files_lane(running(1, "a"))])
        before = {widget for widget in QApplication.topLevelWidgets() if widget.isVisible()}
        for rows in ([running(1, "a"), queued(2, "b", 2, down=False)],
                     [running(1, "a"), queued(2, "b", 2), queued(3, "c", 3, down=False)], [queued(3, "c", 1, up=False, down=False)], []):
            self.source.current = [files_lane(*rows)] if rows else []
            section.refresh()
            flush(self.app)
        self.assertEqual({widget for widget in QApplication.topLevelWidgets() if widget.isVisible()}, before)

    def test_the_second_timer_moves_the_clock_of_a_running_job(self):
        clock_now = [105.0]
        section = self.make([files_lane(running(1, "a"))], now=lambda: clock_now[0])
        clock_now[0] = 172.0
        section.timer.timeout.emit()
        self.assertEqual(section.rows[1].state.full_text(), "Running · 01:12")
        self.assertEqual(section.timer.interval(), 1000)

    def test_the_time_reads_as_minutes_and_seconds_and_hours_after_an_hour(self):
        self.assertEqual([clock(0), clock(5), clock(65), clock(3599), clock(3600), clock(3723), clock(-3)],
                         ["00:00", "00:05", "01:05", "59:59", "1:00:00", "1:02:03", "00:00"])

    def test_keyboard_focus_stays_on_a_moved_jobs_button_when_the_rows_are_rebuilt(self):
        section = self.make([files_lane(running(1, "a"), queued(2, "b", 2, up=False), queued(3, "c", 3, down=False))])
        section.rows[3].up.setFocus(Qt.FocusReason.TabFocusReason)
        flush(self.app)
        self.assertTrue(section.rows[3].up.hasFocus())
        # The scheduler moved job 3 up: the rows are rebuilt in the new order, and the focus follows its button.
        self.source.current = [files_lane(running(1, "a"), queued(3, "c", 2, up=False, down=True),
                                          queued(2, "b", 3, up=True, down=False))]
        section.refresh()
        flush(self.app)
        self.assertEqual(list(section.rows), [1, 3, 2])
        # Job 3 is first in line now, so its Move up button is disabled: the keyboard goes to the button beside it.
        self.assertFalse(section.rows[3].up.isEnabled())
        focused = QApplication.focusWidget()
        self.assertIsNotNone(focused)
        self.assertEqual((focused.property("job"), focused.property("role")), (3, "down"))

    def test_a_mouse_press_gives_the_keyboard_back_and_a_key_press_keeps_it(self):
        section = self.make([files_lane(running(1, "a"), queued(2, "b", 2), queued(3, "c", 3, down=False))])
        back = mock.Mock()
        section.keyboard_back = back
        section.rows[2].down.click()                  # not focused: a click
        back.assert_called_once_with()
        back.reset_mock()
        self.source.current = self.source.current     # rows unchanged
        button = section.rows[2].down
        button.setFocus(Qt.FocusReason.TabFocusReason)
        flush(self.app)
        QTest.keyClick(button, Qt.Key.Key_Space)      # focused: Space presses it
        self.assertEqual(self.source.calls[-1], ("move", 2, 1))
        back.assert_not_called()


class ScheduledJobsCase(SectionCase):
    """The section over the real scheduler, through ViewerJobs."""

    def setUp(self):
        from Background_Job_Scheduler import BackgroundJobScheduler

        class Viewer:
            def set_background_job_status(self, message):
                pass

        self.viewer = Viewer()
        self.scheduler = BackgroundJobScheduler(self.viewer)
        self.viewer.background_job_scheduler = self.scheduler
        self.addCleanup(end_scheduler, self.scheduler, 0.2)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.gate = threading.Event()
        self.addCleanup(self.gate.set)
        self.jobs = ViewerJobs(self.viewer)

    def path(self, name):
        return os.path.join(self.directory.name, name)

    def hold(self, name, lane="files", **options):
        entered = threading.Event()

        def worker(payload):
            entered.set()
            self.gate.wait(10)
            return {}

        job_id = self.scheduler.enqueue("test", f"{name} description", None, worker, self.path(name), lane=lane, **options)
        return job_id, entered

    def wait_for(self, condition, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if condition():
                return True
            time.sleep(0.005)
        self.app.processEvents()
        return condition()


class ViewerJobsTests(ScheduledJobsCase):
    def test_the_lanes_list_the_running_job_first_then_the_waiting_ones_with_what_each_can_do(self):
        first, entered = self.hold("a")
        self.assertTrue(entered.wait(10))
        second, _ = self.hold("b")
        third, _ = self.hold("c")
        fourth, _ = self.hold("d", lane="folding", lane_title=Message("Folding"))
        lanes = self.jobs.lanes()
        self.assertEqual([lane.name for lane in lanes], ["files", "folding"])
        files = lanes[0].rows
        self.assertEqual([(row.job_id, row.state, row.position, row.can_move_up, row.can_move_down) for row in files],
                         [(first, "running", 1, False, False), (second, "queued", 2, False, True),
                          (third, "queued", 3, True, False)])
        self.assertIsNotNone(files[0].started_at)
        self.assertEqual(files[1].description, "b description")
        self.assertEqual(files[1].path, self.path("b"))

    def test_the_badge_lines_name_the_background_jobs_and_nothing_else(self):
        self.assertEqual(job_lines(self.scheduler), [])
        self.assertEqual(job_lines(None), [])
        first, entered = self.hold("a")
        self.assertTrue(entered.wait(10))
        second, _ = self.hold("b")
        self.assertEqual(job_lines(self.scheduler), [f"#{first} running: a description", f"#{second} waiting: b description"])

    def test_a_lane_with_nothing_in_it_is_not_listed(self):
        self.scheduler.register_lane("folding", Message("Folding"))
        self.assertEqual(self.jobs.lanes(), [])

    def test_move_and_remove_act_on_the_scheduler_and_a_stopping_job_is_listed_as_stopping(self):
        first, entered = self.hold("a")
        self.assertTrue(entered.wait(10))
        second, _ = self.hold("b")
        third, _ = self.hold("c")
        self.assertTrue(self.jobs.move(third, -1))
        self.assertEqual([row.job_id for row in self.jobs.lanes()[0].rows], [first, third, second])
        self.assertTrue(self.jobs.remove(second))
        self.assertEqual([row.job_id for row in self.jobs.lanes()[0].rows], [first, third])
        self.assertTrue(self.jobs.remove(first))
        self.assertEqual(self.jobs.lanes()[0].rows[0].state, "stopping")

    def test_a_failed_job_is_listed_with_its_reason_and_log_until_dismissed(self):
        log = self.path("run.log")
        Path(log).write_text("worker output")

        def breaks(payload, job):
            job.set_log_path(log)
            raise RuntimeError(Message("Folding failed: {why}", why="out of memory"))

        with mock.patch("builtins.print"):
            job_id = self.scheduler.enqueue("test", "fold", None, breaks, self.path("f"), lane="folding")
            self.assertTrue(self.wait_for(lambda: bool(self.scheduler.failures())))
        (lane,) = self.jobs.lanes()
        (row,) = lane.rows
        self.assertEqual((row.job_id, row.state, row.log_path), (job_id, "failed", log))
        self.assertTrue(self.jobs.dismiss(job_id))
        self.assertEqual(self.jobs.lanes(), [])

    def test_open_log_opens_the_file_the_job_named_by_its_number_where_a_user_is_at_the_viewer(self):
        log = self.path("run.log")
        Path(log).write_text("worker output")
        job_id, entered = self.hold("a", log_path=log)
        self.assertTrue(entered.wait(10))
        with mock.patch("desktop.Viewer_Jobs.Desktop_Open") as services, \
                mock.patch("desktop.Viewer_Jobs.viewer_shows_windows", return_value=True):
            services.open_path.return_value = True
            self.assertTrue(self.jobs.open_log(job_id))
            opened = services.open_path.call_args.args[0]
            self.assertEqual(os.path.normcase(opened), os.path.normcase(log))
            services.open_path.reset_mock()
            self.assertFalse(self.jobs.open_log(job_id + 100), "a number that names no job opens nothing")
            services.open_path.assert_not_called()

    def test_a_viewer_with_no_windows_and_a_log_that_is_gone_open_nothing(self):
        log = self.path("run.log")
        job_id, entered = self.hold("a", log_path=log)
        self.assertTrue(entered.wait(10))
        with mock.patch("desktop.Viewer_Jobs.Desktop_Open") as services, \
                mock.patch("desktop.Viewer_Jobs.viewer_shows_windows", return_value=True):
            self.assertFalse(self.jobs.open_log(job_id), "the file is not there")
            Path(log).write_text("x")
            with mock.patch("desktop.Viewer_Jobs.viewer_shows_windows", return_value=False):
                self.assertFalse(self.jobs.open_log(job_id), "nobody is at the Viewer")
            services.open_path.assert_not_called()


class JobsBesideCommandsTests(ScheduledJobsCase):
    """Cluster and the commands that wait behind it are not background jobs: the lists and the numbers leave them out."""

    def setUp(self):
        super().setUp()
        import io
        from contextlib import redirect_stdout
        from tests.test_command_queue import QueueViewer
        self.queue_viewer = QueueViewer()
        self.addCleanup(self.queue_viewer.close)
        self.addCleanup(self.queue_viewer.command_queue.shutdown)
        self.queue_viewer.background_job_scheduler = self.scheduler
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_the_lists_hold_no_command_and_the_first_job_is_number_one(self):
        from tests.test_command_queue import CLUSTER
        self.queue_viewer.type(CLUSTER)
        self.queue_viewer.type("color red")
        self.assertEqual(len(self.queue_viewer.command_queue.pending()), 2)
        self.assertEqual((self.jobs.lanes(), job_lines(self.scheduler)), ([], []))
        first, entered = self.hold("a")
        self.assertTrue(entered.wait(10))
        self.assertEqual(first, 1)
        (lane,) = self.jobs.lanes()
        self.assertEqual([(row.job_id, row.state) for row in lane.rows], [(1, "running")])
        self.assertEqual(job_lines(self.scheduler), ["#1 running: a description"])
        self.assertEqual(len(self.queue_viewer.command_queue.pending()), 2)


class SectionOverSchedulerTests(ScheduledJobsCase):
    def test_the_list_follows_the_scheduler_through_its_changed_signal(self):
        section = JobsSection(self.jobs)
        self.addCleanup(section.deleteLater)
        self.scheduler.changed.connect(section.refresh)
        section.show()
        self.assertTrue(section.empty.isVisible())
        first, entered = self.hold("a")
        self.assertTrue(entered.wait(10))
        self.assertTrue(self.wait_for(lambda: 1 in section.rows and section.rows[1].row.state == "running"))
        second, _ = self.hold("b")
        self.assertTrue(self.wait_for(lambda: second in section.rows))
        self.assertEqual(section.rows[second].state.full_text(), "Queued · position 2")
        section.rows[second].remove.click()
        self.assertTrue(self.wait_for(lambda: second not in section.rows))
        self.gate.set()
        self.assertTrue(self.wait_for(lambda: section.empty.isVisible()))

    def test_stopping_a_running_job_from_the_section_stops_it_after_the_confirmation(self):
        stopped = threading.Event()
        entered = threading.Event()

        def worker(payload, job):
            entered.set()
            while not job.requested:
                time.sleep(0.005)
            stopped.set()
            job.raise_if_cancelled()

        section = JobsSection(self.jobs, confirm=lambda job, description: True)
        self.addCleanup(section.deleteLater)
        self.scheduler.changed.connect(section.refresh)
        section.show()
        job_id = self.scheduler.enqueue("test", "wrapper", None, worker, self.path("w"))
        self.assertTrue(entered.wait(10))
        self.assertTrue(self.wait_for(lambda: job_id in section.rows and section.rows[job_id].row.state == "running"))
        section.rows[job_id].remove.click()
        self.assertTrue(stopped.wait(10))
        self.assertTrue(self.wait_for(lambda: section.empty.isVisible()))


class CardTests(ScheduledJobsCase):
    """The card holds the section between the files and the controls, and follows it as the card is open."""

    def setUp(self):
        super().setUp()
        self.host = RecordingCanvas()
        self.addCleanup(dispose, self.host)
        self.host.setGeometry(10, 0, 1000, 800)
        self.info = {"node_count": 10, "edge_count": 20, "inputs": {}, "directories": {}, "project_root": ""}
        self.button = ControlsButton(
            "Session and controls", shipped_actions(), self.host, host=self.host, session=lambda: self.info,
            controls_title="Mouse and keyboard controls",
            area=lambda: self.host.rect().adjusted(8, 8, -8, -8), command=mock.Mock(), jobs=self.jobs,
            confirm_stop=lambda job, description: True)
        self.button.move(20, 760)
        self.host.show()
        self.host.activateWindow()
        QTest.qWaitForWindowActive(self.host)
        flush(self.app)
        self.scheduler.changed.connect(self.button.refresh_card_jobs)

    def open_card(self):
        card = self.button.toggle_card()
        flush(self.app)
        self.assertTrue(card.isVisible())
        return card

    def test_a_card_opened_over_jobs_that_are_already_there_fits_them_without_scrolling(self):
        # The rows are made before the card is styled and shown, and size themselves as they are: the card
        # measures again once they have, or its last rows would be cut off and need a scroll bar.
        self.host.setGeometry(10, 0, 1000, 1600)
        self.button.move(20, 1560)
        first, entered = self.hold("a")
        self.assertTrue(entered.wait(10))
        self.hold("b")
        self.hold("c", lane="folding", lane_title=Message("Folding"))
        card = self.open_card()
        QTest.qWait(150)   # the card measures again on the event loop's next turn, long before a clock tick
        flush(self.app)
        self.assertEqual(card.scroll.verticalScrollBar().maximum(), 0, "nothing is left to scroll to")
        self.assertEqual(card.scroll.horizontalScrollBar().maximum(), 0)

    def test_the_open_log_link_keeps_the_height_its_text_needs_inside_the_card(self):
        # The card's stylesheet shapes the link: a rule that lets the layout squeeze it cuts the text off below.
        log = self.path("fold.log")
        Path(log).write_text("worker output", encoding="utf-8")

        def worker(payload, job):
            job.set_log_path(log)
            raise RuntimeError("it failed")

        self.scheduler.enqueue("test", "fold", None, worker, self.path("fold"))
        self.assertTrue(self.wait_for(lambda: bool(self.scheduler.failures())))
        card = self.open_card()
        flush(self.app)
        [row] = card.jobs_section.rows.values()
        link = row.log_button
        self.assertEqual(link.objectName(), "jobLogLink")
        self.assertGreaterEqual(link.height(), link.sizeHint().height(), "the link is as tall as its text and padding")
        self.assertGreaterEqual(link.minimumHeight(), link.sizeHint().height() - 1, "and may not be made shorter")

    def test_the_section_sits_between_the_files_and_the_controls_each_set_off_by_a_separator(self):
        card = self.open_card()
        section = card.jobs_section
        body = card.body
        top = lambda widget: widget.mapTo(body, QPoint(0, 0)).y()
        embedding = card.path_fields["embedding"]
        self.assertLess(top(embedding), top(section.heading))
        self.assertLess(top(section.heading), top(card.controls_heading))
        separators = sorted(top(frame) for frame in card.findChildren(type(card._separator())) if frame.objectName() == "controlsSeparator")
        self.assertEqual(len(separators), 2)
        self.assertLess(top(embedding), separators[0])
        self.assertLess(separators[0], top(section.heading))
        self.assertLess(top(section.heading), separators[1])
        self.assertLess(separators[1], top(card.controls_heading))

    def test_with_no_job_the_card_holds_the_muted_line_and_a_card_with_no_source_holds_no_section(self):
        card = self.open_card()
        self.assertEqual(card.jobs_section.empty.text(), "No background jobs.")
        card.close()
        plain = ControlsButton("Session and controls", shipped_actions(), self.host, host=self.host, session=lambda: self.info,
                               controls_title="Mouse and keyboard controls")
        plain.move(20, 760)
        plain.show()
        card = plain.toggle_card()
        self.assertIsNone(card.jobs_section)
        card.close()

    def test_the_open_card_grows_upward_as_jobs_arrive_and_shrinks_when_they_are_gone(self):
        card = self.open_card()
        empty_height, bottom = card.height(), card.y() + card.height()
        first, entered = self.hold("a")
        self.assertTrue(entered.wait(10))
        second, _ = self.hold("b")
        third, _ = self.hold("c")
        self.assertTrue(self.wait_for(lambda: len(card.jobs_section.rows) == 3))
        flush(self.app)
        self.assertGreater(card.height(), empty_height)
        self.assertEqual(card.y() + card.height(), bottom, "the card stays above its button")
        self.assertGreaterEqual(card.y(), 0)
        self.gate.set()
        self.assertTrue(self.wait_for(lambda: card.jobs_section.empty.isVisible()))
        flush(self.app)
        self.assertEqual(card.height(), empty_height)

    def test_new_jobs_never_take_the_keyboard_from_the_canvas(self):
        card = self.open_card()
        self.host.setFocus()
        flush(self.app)
        self.assertIs(QApplication.focusWidget(), self.host)
        first, entered = self.hold("a")
        self.assertTrue(entered.wait(10))
        self.hold("b")
        self.hold("c")
        self.assertTrue(self.wait_for(lambda: len(card.jobs_section.rows) == 3))
        flush(self.app)
        self.assertIs(QApplication.focusWidget(), self.host, "[ENTER] still reaches the canvas")

    def test_a_closed_card_costs_nothing_when_jobs_change(self):
        self.button.refresh_card_jobs()
        first, entered = self.hold("a")
        self.assertTrue(entered.wait(10))
        self.assertTrue(self.wait_for(lambda: bool(self.scheduler.snapshot())))
        self.assertFalse(self.button.card_is_open())

    def test_tab_from_the_canvas_reaches_the_jobs_buttons_and_a_click_does_not_take_the_keyboard(self):
        card = self.open_card()
        first, entered = self.hold("a")
        self.assertTrue(entered.wait(10))
        second, _ = self.hold("b")
        third, _ = self.hold("c")
        self.assertTrue(self.wait_for(lambda: len(card.jobs_section.rows) == 3))
        flush(self.app)
        self.host.setFocus()
        flush(self.app)
        self.assertIs(QApplication.focusWidget(), self.host)
        QTest.keyClick(self.host, Qt.Key.Key_Tab)
        flush(self.app)
        focused = QApplication.focusWidget()
        self.assertIsNotNone(focused)
        self.assertIn(focused.property("role"), ("remove", "up", "down"), "Tab reaches a jobs button")
        self.assertTrue(card.jobs_section.isAncestorOf(focused))
        # A click on a button does not keep the keyboard.
        self.host.setFocus()
        flush(self.app)
        card.jobs_section.rows[third].up.click()
        flush(self.app)
        self.assertIs(QApplication.focusWidget(), self.host)


class ViewerWiringTests(unittest.TestCase):
    """The Viewer gives the card its jobs and tells the card when the scheduler changes (read from its source)."""

    SOURCE = (SRC / "EMAPSSN_Viewer.py").read_text(encoding="utf-8")

    def test_the_card_is_given_the_scheduler_s_jobs(self):
        self.assertIn("jobs=ViewerJobs(self),", self.SOURCE)

    def test_one_changed_signal_refreshes_the_badge_and_an_open_card(self):
        self.assertIn("self.background_job_scheduler.changed.connect(self._on_background_jobs_changed)", self.SOURCE)
        handler = self.SOURCE[self.SOURCE.index("def _on_background_jobs_changed"):]
        handler = handler[:handler.index("def _refresh_jobs_badge")]
        self.assertIn("self._refresh_jobs_badge()", handler)
        self.assertIn("button.refresh_card_jobs()", handler)


class SchedulerCardSourcesNameNoCommandTests(unittest.TestCase):
    """The card's own files know no command either (the same check as the scheduler's)."""

    def test_the_card_and_its_rows_name_no_command(self):
        from tests.test_background_job_scheduler import SchedulerNamesNoCommandTests

        checker = SchedulerNamesNoCommandTests()
        for relative in ("desktop/Viewer_Jobs.py", "desktop/Viewer_Jobs_Card.py"):
            with self.subTest(file=relative):
                self.assertEqual(checker.mentions(relative), [])


if __name__ == "__main__":
    unittest.main()
