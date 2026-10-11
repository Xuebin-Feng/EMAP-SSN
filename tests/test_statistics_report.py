# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The statistics report (desktop.Desktop_App.StatisticsReport) as the Config windows show it.

Its threshold table draws, behind each threshold, a bar for the edges from
that threshold up to the next (the counts are of edges at or above each
threshold), and keeps the count and percentage columns to their text's width
while the report is wide, so the threshold column, and its bars, take the rest.
"""

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtGui import QColor, QTextLength  # noqa: E402

from desktop.Desktop_App import StatisticsReport, qt_monospace_font  # noqa: E402
from desktop.Studio_Theme import TOKENS, set_role  # noqa: E402
from tests.config_gui_loader import load_config_namespace  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402

SUMMARY = ["Network Model: E1_RA", "Total Nodes Processed: 12927"]
HEADINGS = ("Threshold", "Count", "Percentage")
# Cumulative counts; their buckets are 400, 100, 300 and 0.
ROWS = [("0.4", "800", "100.00%"), ("0.5", "400", "50.00%"), ("0.6", "300", "37.50%"), ("0.7", "0", "0.00%")]


def flush(app):
    for _ in range(6):
        app.processEvents()


class StatisticsReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = load_config_namespace()
        cls.app = namespace["QApplication"].instance() or namespace["QApplication"]([])
        namespace["configure_qt_application_fonts"](cls.app)
        cls.report_html = staticmethod(namespace["statistics_report_html"])
        apply_theme_for_class(cls, cls.app)

    def open_report(self, width):
        report = StatisticsReport()
        self.addCleanup(report.deleteLater)
        report.setFont(qt_monospace_font(report.font()))
        set_role(report, "report")
        report.setHtml(self.report_html(SUMMARY, HEADINGS, ROWS))
        report.resize(width, 400)
        report.show()
        flush(self.app)
        return report

    def first_cell(self, report, row):
        table = report._tables()[0]
        block = table.cellAt(row, 0).firstCursorPosition().block()
        return report.document().documentLayout().blockBoundingRect(block)

    def test_each_row_gets_a_bar_for_the_edges_up_to_the_next_threshold(self):
        report = self.open_report(700)
        self.assertEqual([bucket for _, _, bucket in report.bar_values()], [400, 100, 300, 0])
        bars = report.bar_rects()
        # The row with no edges has no bar.
        self.assertEqual(len(bars), 3)
        cell = self.first_cell(report, 1)
        self.assertAlmostEqual(bars[0].left(), cell.left())
        self.assertAlmostEqual(bars[0].width(), cell.width())
        self.assertAlmostEqual(bars[1].width(), cell.width() / 4)
        self.assertAlmostEqual(bars[2].width(), cell.width() * 3 / 4)
        for row, bar in enumerate(bars, start=1):
            with self.subTest(row=row):
                cell = self.first_cell(report, row)
                self.assertGreaterEqual(bar.top(), cell.top())
                self.assertLessEqual(bar.bottom(), cell.bottom())

    def test_a_wide_report_keeps_count_and_percentage_to_their_text(self):
        report = self.open_report(800)
        constraints = report._tables()[0].format().columnWidthConstraints()
        self.assertEqual([length.type() for length in constraints], [
            QTextLength.Type.VariableLength, QTextLength.Type.FixedLength, QTextLength.Type.FixedLength,
        ])
        digit = report.fontMetrics().horizontalAdvance("0")
        for length in constraints[1:]:
            # The heading or the widest figure, the gap before it, and a few characters more.
            self.assertLess(length.rawValue(), 20 * digit)
        self.assertGreater(self.first_cell(report, 1).width(), report.viewport().width() / 2)

    def test_a_narrow_report_lays_the_table_out_as_qt_does(self):
        # As narrow as Config's report gets, beside the smallest window's settings.
        report = self.open_report(260)
        self.assertEqual(report._tables()[0].format().columnWidthConstraints(), [])
        self.assertEqual(len(report.bar_rects()), 3)
        self.assertLessEqual(report.document().size().width(), report.viewport().width() + 1)

    def test_the_bars_are_drawn_behind_the_text_in_the_histogram_colour(self):
        report = self.open_report(700)
        bar = report.bar_rects()[0]
        image = report.viewport().grab().toImage()
        # Past the threshold's text, in the middle of the longest bar.
        point = bar.center().toPoint()
        point.setX(int(bar.left() + bar.width() * 0.8))
        point -= report.viewport().mapFrom(report.viewport(), point) - point
        self.assertEqual(image.pixelColor(point).name(), QColor(TOKENS["histogram"]).name())

    def test_the_bars_come_back_with_the_report_after_a_redraw(self):
        report = self.open_report(700)
        copy = self.open_report(700)
        copy.setHtml(report.toHtml())
        flush(self.app)
        self.assertEqual(len(copy.bar_rects()), 3)
        plain = self.open_report(700)
        plain.setPlainText("Select Fasta subset and HDF5 Network file, then click compute.")
        flush(self.app)
        self.assertEqual(plain.bar_rects(), [])


if __name__ == "__main__":
    unittest.main()
