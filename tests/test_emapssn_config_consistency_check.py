# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Configuration window: the Consistency Check report.

The check reads FASTA, network and MSA headers with the canonical sanitization,
reports whether the FASTA is a subset of the network and how much of it the MSA
and the embedding cover, and resolves the alignment reference as the Viewer
does. The report is a table with a row per check, in the tip panel; what a
check found wrong follows the table, one paragraph for each.
"""

import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import EMAPSSN_Config  # noqa: E402
from desktop.Studio_Theme import TOKENS, status_color  # noqa: E402
from tests.config_gui_loader import load_config_namespace, open_config_window  # noqa: E402
from tests.consistency_report import (  # noqa: E402
    ZERO_WIDTH_SPACE,
    clean,
    lines_per_cell,
    parse_report,
)
from tests.embedding_fixtures import write_embedding_file  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402

_GUI_CLASS = None
_APP = None
_NAMESPACE = None


def setUpModule():
    # run_consistency_check needs only the class; no window (and no settings) is opened.
    global _GUI_CLASS, _APP, _NAMESPACE
    _NAMESPACE = load_config_namespace()
    _APP = _NAMESPACE["QApplication"].instance() or _NAMESPACE["QApplication"]([])
    _GUI_CLASS = _NAMESPACE["ConfigGUI"]


class ConsistencyHeaderSanitizationTests(unittest.TestCase):
    def test_fasta_headers_use_canonical_sanitization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = pathlib.Path(temp_dir) / "sequences.fasta"
            fasta_path.write_text(
                ">  Alpha   Beta  \nac-d\n",
                encoding="utf-8",
            )

            headers = EMAPSSN_Config._load_consistency_fasta_headers(fasta_path)

        self.assertEqual(headers, ["Alpha_Beta"])

    def test_fasta_and_sparse_msa_headers_share_the_canonical_rule(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            msa_fasta_path = temp_path / "alignment.fasta"
            msa_hdf5_path = temp_path / "alignment.h5"
            msa_fasta_path.write_text(
                ">Alpha   Beta\nAC-D\n",
                encoding="utf-8",
            )
            with h5py.File(msa_hdf5_path, "w") as hf:
                hf.create_dataset("headers", data=[b"Alpha   Beta"])

            fasta_headers = EMAPSSN_Config._load_consistency_msa_headers(
                msa_fasta_path
            )
            hdf5_headers = EMAPSSN_Config._load_consistency_msa_headers(
                msa_hdf5_path
            )

        self.assertEqual(fasta_headers, ["Alpha_Beta"])
        self.assertEqual(hdf5_headers, ["Alpha_Beta"])


class _Choice:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text

    def text(self):
        return self._text


class _Panel:
    """Stands for the tip panel: keeps the report it was given, or the plain message."""

    report_html = ""
    message = ""

    def setText(self, text):
        self.message, self.report_html = text, ""

    def setReport(self, report_html):
        self.message, self.report_html = "", report_html


class _Report:
    """What a Consistency Check showed: its table's rows and the paragraphs below it."""

    def __init__(self, panel):
        self.html = panel.report_html
        parsed = parse_report(panel.report_html)
        self.rows, self.notes = parsed.rows, parsed.notes

    def row(self, name):
        found = [row for row in self.rows if row.name == name]
        assert len(found) == 1, (name, [row.name for row in self.rows])
        return found[0]

    def has_row(self, name):
        return any(row.name == name for row in self.rows)

    def notes_of(self, name):
        """(colour, text) of each paragraph below the table that opens with a check's name."""
        return [(colour, text[len(name):].strip()) for colour, text in self.notes
                if text.startswith(name + " ")]


class _ReportFixture:
    """A FASTA, network and MSA in a folder, and the report a Consistency Check gives."""

    # Distinct sequences: the canonical FASTA reader drops exact duplicates.
    RECORDS = {
        "WP_0123.1_protein_A": "MKTAYIAK",
        "WP_0123.10_protein_B": "MSEQNNTE",
        "Other_seq": "MAAAGGGK",
    }

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        folder = pathlib.Path(self._temp.name)
        (folder / "nodes.fasta").write_text(
            "".join(f">{header}\n{sequence}\n" for header, sequence in self.RECORDS.items()),
            encoding="utf-8",
        )
        with h5py.File(folder / "network.h5", "w") as hf:
            hf.create_dataset("headers", data=[header.encode() for header in self.RECORDS])
        # The MSA lacks WP_0123.10_protein_B.
        (folder / "alignment.fasta").write_text(
            ">WP_0123.1_protein_A\nMKTAYIAK\n>Other_seq\nMAAAGGGK\n", encoding="utf-8"
        )
        self.folder = str(folder)

    def tearDown(self):
        self._temp.cleanup()

    def run_check(self, reference, fasta="nodes.fasta", embedding="", network="network.h5",
                  msa="alignment.fasta"):
        panel = _Panel()
        stand_in = SimpleNamespace(
            cb_fasta=_Choice(fasta),
            cb_hdf5=_Choice(network),
            cb_msa=_Choice(msa),
            cb_embed=_Choice(embedding),
            line_ref=_Choice(reference),
            tip_panel=panel,
            _resolved_directory_input=lambda key: self.folder,
        )
        _GUI_CLASS.run_consistency_check(stand_in)
        return panel

    def report(self, reference, **choices):
        report = _Report(self.run_check(reference, **choices))
        self.assertTrue(report.rows, report.notes)
        self.assertFalse(
            [text for _, text in report.notes if "Error during consistency check" in text]
        )
        return report


class ConsistencyReferenceReportTests(_ReportFixture, unittest.TestCase):
    """The report resolves the reference as the Viewer does, over the network headers."""

    def test_a_leading_identifier_names_one_header_not_every_substring_match(self):
        report = self.report("WP_0123.1")
        row = report.row("Reference ID")
        self.assertEqual(row.result, "'WP_0123.1' resolves to WP_0123.1_protein_A")
        self.assertEqual(row.mark_colour, status_color("success"))
        self.assertEqual(report.notes_of("Reference ID"), [])

    def test_tied_matches_name_the_header_used_and_the_others(self):
        report = self.report("WP_01*")
        row = report.row("Reference ID")
        self.assertEqual(row.result, "'WP_01*' resolves to WP_0123.1_protein_A")
        # A remark, not a problem: the row keeps its success mark.
        self.assertEqual(row.mark_colour, status_color("success"))
        self.assertEqual(report.notes_of("Reference ID"), [(
            status_color("subtle"),
            "It matches 2 headers equally; the first in network order is used. "
            "Others: WP_0123.10_protein_B",
        )])

    def test_a_wildcard_is_resolved_rather_than_reported_missing(self):
        row = self.report("*.1").row("Reference ID")
        self.assertEqual(row.result, "'*.1' resolves to WP_0123.1_protein_A")
        self.assertEqual(row.mark_colour, status_color("success"))

    def test_a_resolved_header_the_msa_lacks_is_named(self):
        report = self.report("WP_0123.10")
        row = report.row("Reference ID")
        self.assertEqual(row.result, "'WP_0123.10' resolves to WP_0123.10_protein_B")
        self.assertEqual(row.mark_colour, status_color("warning"))
        self.assertEqual(report.notes_of("Reference ID"), [(
            status_color("warning"),
            "The MSA lacks this header, so positions will be numbered by occupancy and "
            "the offset ignored.",
        )])

    def test_an_unmatched_reference_leaves_numbering_inactive(self):
        row = self.report("XYZ").row("Reference ID")
        self.assertEqual(
            row.result,
            "'XYZ' matches no network header, so reference numbering will be inactive.",
        )
        self.assertEqual(row.mark_colour, status_color("warning"))
        self.assertEqual(row.span, 2)

    def test_no_reference_adds_no_row(self):
        self.assertFalse(self.report("").has_row("Reference ID"))

    def test_subset_success_and_incomplete_msa_coverage_are_reported(self):
        report = self.report("")
        network, msa = report.rows
        self.assertEqual(
            (network.name, network.result, network.details, network.mark_colour),
            ("FASTA vs HDF5", "3 of 3 matched", ["strict subset"], status_color("success")),
        )
        self.assertEqual(
            (msa.name, msa.result, msa.details, msa.mark_colour),
            ("FASTA vs MSA", "2 of 3 matched", ["coverage incomplete: 1 missing"],
             status_color("warning")),
        )
        self.assertEqual(report.notes, [(
            status_color("warning"),
            "FASTA vs MSA Missing examples: WP_0123.10_protein_B\n"
            "Missing nodes remain plotted but are excluded from alignment-dependent "
            "analyses.",
        )])

    def test_fasta_headers_absent_from_the_network_are_an_error(self):
        (pathlib.Path(self.folder) / "superset.fasta").write_text(
            "".join(f">{header}\n{sequence}\n" for header, sequence in self.RECORDS.items())
            + ">Extra seq\nMCCCCCCK\n",
            encoding="utf-8",
        )
        report = self.report("", fasta="superset.fasta")
        network, msa = report.rows
        self.assertEqual(
            (network.result, network.details, network.mark_colour),
            ("3 of 4 matched", ["NOT a subset: 1 missing"], status_color("danger")),
        )
        self.assertEqual(report.notes_of("FASTA vs HDF5"), [
            (status_color("danger"), "Missing examples: Extra_seq"),
        ])
        self.assertEqual(
            (msa.result, msa.details), ("2 of 4 matched", ["coverage incomplete: 2 missing"])
        )
        self.assertEqual(report.notes_of("FASTA vs MSA")[0][1].splitlines()[0],
                         "Missing examples: WP_0123.10_protein_B, Extra_seq")

    def test_an_unreadable_network_is_reported_as_an_error_without_a_table(self):
        (pathlib.Path(self.folder) / "broken.h5").write_text("not an HDF5 file", encoding="utf-8")
        report = _Report(self.run_check("", network="broken.h5"))
        self.assertEqual(report.rows, [])
        (colour, text), = report.notes
        self.assertEqual(colour, status_color("danger"))
        self.assertTrue(text.startswith("Error during consistency check: "), text)


class ConsistencyEmbeddingReportTests(_ReportFixture, unittest.TestCase):
    """The embedding row matches nodes by header and sequence, as Embedding_Manager does."""

    NAME = "FASTA vs embedding"

    def embedding_report(self, records, *, model_name="test_model", network_model=None):
        folder = pathlib.Path(self.folder)
        write_embedding_file(folder / "set_embeddings.h5", records, model_name=model_name)
        if network_model is not None:
            with h5py.File(folder / "network.h5", "a") as hf:
                hf.attrs["model_name"] = network_model
        return self.report("", embedding="set_embeddings.h5")

    def test_full_coverage_is_a_success_naming_the_model(self):
        row = self.embedding_report(list(self.RECORDS.items())).row(self.NAME)
        self.assertEqual(row.result, "3 of 3 matched")
        self.assertEqual(
            row.details,
            ["covers all FASTA nodes", "Embedding model: test_model (width 3, float16)."],
        )
        self.assertEqual(row.mark_colour, status_color("success"))

    def test_missing_and_different_sequences_are_a_warning(self):
        records = [("WP_0123.1_protein_A", "MKTAYIAK"), ("Other_seq", "MAAAGGG")]
        report = self.embedding_report(records)
        row = report.row(self.NAME)
        self.assertEqual(row.result, "1 of 3 matched")
        self.assertEqual(
            row.details,
            ["coverage incomplete: 1 missing, 1 different sequence",
             "Embedding model: test_model (width 3, float16)."],
        )
        self.assertEqual(row.mark_colour, status_color("warning"))
        self.assertEqual(report.notes_of(self.NAME), [(
            status_color("warning"),
            "Missing examples: WP_0123.10_protein_B\n"
            "Examples with a different sequence: Other_seq\n"
            "Missing nodes, and nodes whose sequence differs, remain plotted but are "
            "excluded from embedding-dependent analyses.",
        )])

    def test_a_network_built_with_another_model_gets_a_note(self):
        report = self.embedding_report(
            list(self.RECORDS.items()), model_name="esmc_6b", network_model="ankh_base"
        )
        # A remark under an unchanged success row.
        self.assertEqual(report.row(self.NAME).mark_colour, status_color("success"))
        self.assertEqual(report.notes_of(self.NAME), [(
            status_color("subtle"),
            "NOTE: The network was built with ankh_base, not esmc_6b. The Viewer accepts "
            "this; embedding commands use the embedding's model.",
        )])
        same = self.embedding_report(
            list(self.RECORDS.items()), model_name="esmc_6b", network_model="esmc_6b"
        )
        self.assertEqual(same.notes_of(self.NAME), [])
        # A BLAST network was built from no embedding, so there is nothing to compare.
        blast = self.embedding_report(
            list(self.RECORDS.items()), model_name="esmc_6b", network_model="BLAST"
        )
        self.assertEqual(blast.notes_of(self.NAME), [])

    def test_a_file_the_viewer_would_refuse_is_an_error(self):
        folder = pathlib.Path(self.folder)
        write_embedding_file(folder / "set_embeddings.h5", list(self.RECORDS.items()),
                             complete=False)
        report = self.report("", embedding="set_embeddings.h5")
        row = report.row(self.NAME)
        self.assertEqual(row.mark_colour, status_color("danger"))
        self.assertEqual(row.result, "the file cannot be used")
        self.assertEqual(report.notes_of(self.NAME), [(
            status_color("danger"),
            "The embedding file cannot be used, so the Viewer will not open with it: "
            "Embedding generation is incomplete.",
        )])

    def test_no_embedding_adds_no_row(self):
        self.assertFalse(self.report("").has_row(self.NAME))


class ConsistencyReportTableTests(_ReportFixture, unittest.TestCase):
    """One row per check, a mark that differs by shape, and the theme's colours."""

    @classmethod
    def setUpClass(cls):
        # The text's width depends on the font, so use the program's.
        _NAMESPACE["configure_qt_application_fonts"](_APP)
        apply_theme_for_class(cls, _APP)

    def full_report(self, reference="WP_0123.1"):
        folder = pathlib.Path(self.folder)
        write_embedding_file(folder / "set_embeddings.h5", list(self.RECORDS.items()),
                             model_name="esmc_6b", width=2560)
        # The MSA of this report holds every node, so every check succeeds.
        (folder / "alignment.fasta").write_text(
            "".join(f">{header}\n{sequence}\n" for header, sequence in self.RECORDS.items()),
            encoding="utf-8",
        )
        return self.report(reference, embedding="set_embeddings.h5")

    def test_every_check_is_one_row_and_a_success_adds_nothing_below_the_table(self):
        report = self.full_report()
        self.assertEqual(
            [row.name for row in report.rows],
            ["FASTA vs HDF5", "FASTA vs MSA", "FASTA vs embedding", "Reference ID"],
        )
        self.assertEqual(report.notes, [])
        # What the old report said in three or four lines each: the verdict, the counts and,
        # for the embedding, its model, all in the row.
        self.assertEqual(
            [(row.result, row.details) for row in report.rows[:3]],
            [("3 of 3 matched", ["strict subset"]),
             ("3 of 3 matched", ["covers all FASTA nodes"]),
             ("3 of 3 matched", ["covers all FASTA nodes",
                                 "Embedding model: esmc_6b (width 2560, float16)."])],
        )
        self.assertEqual(report.rows[3].result, "'WP_0123.1' resolves to WP_0123.1_protein_A")
        self.assertEqual({row.mark_colour for row in report.rows}, {status_color("success")})

    def test_the_details_take_the_width_the_other_columns_leave(self):
        # The report of a real 12,927-node network, with a long reference
        # header under the table's other rows. At the text width of the default
        # window's panel (654 px) the embedding's two lines, the verdict and its
        # model, stay two lines: the details are not squeezed into a share of
        # the width.
        check = _NAMESPACE["ConsistencyCheck"]
        reference = ("sid|18367|pid|10722|hfid|10|sfid|2|gi|914684625|pdb|3WWH(A)|"
                     "taxonID|1042534|Arthrobactersp.KNK168")
        report_html = _NAMESPACE["consistency_report_html"]([
            check("success", "FASTA vs HDF5", "12927 of 12927 matched", ["strict subset"]),
            check("success", "FASTA vs MSA", "12927 of 12927 matched", ["covers all FASTA nodes"]),
            check("success", "FASTA vs embedding", "12927 of 12927 matched",
                  ["covers all FASTA nodes", "Embedding model: esmc_6b (width 2560, float16)."]),
            check("success", "Reference ID", f"'3WWH' resolves to {reference}"),
        ])
        rows = lines_per_cell(report_html, 654)
        self.assertEqual(rows[:3], [(1, 1, 1), (1, 1, 1), (1, 1, 2)])
        self.assertEqual(rows[3], (1, 2, None))

    def test_a_status_is_told_by_the_shape_of_its_mark_as_well_as_its_colour(self):
        mark = _NAMESPACE["consistency_mark"]
        marks = {kind: mark(kind) for kind in ("success", "warning", "danger", "subtle")}
        self.assertEqual(len(set(marks.values())), 4)

    def test_status_colours_come_from_the_theme(self):
        colours = {"success": "#010203", "warning": "#040506", "danger": "#070809",
                   "text_subtle": "#0a0b0c", "text_muted": "#0d0e0f"}
        with mock.patch.dict(TOKENS, colours):
            report = self.report("XYZ")
            tie = self.report("WP_01*")
            error = _Report(self.run_check("", network="missing.h5"))
        network, msa, reference = report.rows
        self.assertEqual(network.mark_colour, "#010203")
        self.assertEqual(network.verdict_colour, "#0d0e0f")  # a success's verdict is quiet
        self.assertEqual((msa.mark_colour, msa.verdict_colour), ("#040506", "#040506"))
        self.assertEqual(reference.mark_colour, "#040506")
        self.assertEqual([colour for colour, _ in report.notes], ["#040506"])
        # A remark that changes no status takes the subtle colour, and an error the danger one.
        self.assertEqual([colour for colour, _ in tie.notes_of("Reference ID")], ["#0a0b0c"])
        self.assertEqual([colour for colour, _ in error.notes], ["#070809"])
        # Nothing is left over from another theme: the colours follow TOKENS again.
        self.assertEqual(self.report("XYZ").rows[0].mark_colour, status_color("success"))

    def test_text_that_looks_like_markup_is_shown_as_text(self):
        report = self.report("<b>x</b>&amp;")
        self.assertEqual(
            report.row("Reference ID").result,
            "'<b>x</b>&amp;' matches no network header, so reference numbering will be inactive.",
        )

    def test_a_long_unbroken_header_gets_places_to_wrap(self):
        wrappable = _NAMESPACE["wrappable"]
        header = "tr_" + "A0A0A0_9BACT_Uncharacterized_protein_" * 5 + "end"
        shown = wrappable(header)
        self.assertGreater(shown.count(ZERO_WIDTH_SPACE), 10)
        self.assertEqual(clean(shown), header)
        # Every stretch between break places is short, whatever the separators.
        self.assertLessEqual(max(map(len, shown.split(ZERO_WIDTH_SPACE))), 21)
        self.assertEqual(wrappable("A" * 200).count(ZERO_WIDTH_SPACE), 9)
        # Ordinary words, headers and sentences are left alone.
        sentence = "Missing nodes remain plotted but are excluded from alignment-dependent analyses."
        self.assertEqual(wrappable(sentence), sentence)
        self.assertEqual(wrappable("WP_012900.1_hypothetical_protein"), "WP_012900.1_hypothetical_protein")


class ConsistencyReportWindowTests(unittest.TestCase):
    """The report in the window: the whole of it shows, and it never widens the window."""

    @classmethod
    def setUpClass(cls):
        cls.app = _NAMESPACE["QApplication"].instance() or _NAMESPACE["QApplication"]([])
        _NAMESPACE["configure_qt_application_fonts"](cls.app)
        apply_theme_for_class(cls, cls.app)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = pathlib.Path(directory.name)
        self.window = open_config_window(_GUI_CLASS, self.root)
        self.addCleanup(self.close_window)
        self.window.resize(1000, 650)
        self.window.show()
        self.flush()

    def close_window(self):
        self.window.close()
        self.window.deleteLater()
        self.flush()

    def flush(self):
        for _ in range(4):
            self.app.processEvents()

    def run_check(self, headers, reference, *, msa_lacks=1):
        """Check a FASTA of headers against a network and an MSA that lacks the last of them."""
        window = self.window
        work = self.root / "work"
        work.mkdir(exist_ok=True)
        alphabet = "ACDEFGHIKLMNPQRSTVWY"

        def sequence(index):
            digits = []
            for _ in range(5):
                digits.append(alphabet[index % 20])
                index //= 20
            return "M" + "".join(digits)

        records = {header: sequence(index) for index, header in enumerate(headers)}
        (work / "nodes.fasta").write_text(
            "".join(f">{header}\n{text}\n" for header, text in records.items()), encoding="utf-8")
        (work / "alignment.fasta").write_text(
            "".join(f">{header}\n{text}\n" for header, text in list(records.items())[:-msa_lacks]),
            encoding="utf-8")
        with h5py.File(work / "network.h5", "w") as network:
            network.create_dataset("headers", data=[header.encode() for header in headers])
        for combo, name in ((window.cb_fasta, "nodes.fasta"), (window.cb_hdf5, "network.h5"),
                            (window.cb_msa, "alignment.fasta")):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(name)
            combo.setCurrentText(name)
            combo.blockSignals(False)
        window.line_ref.setText(reference)
        with mock.patch.object(window, "_resolved_directory_input", return_value=str(work)):
            window.run_consistency_check()
        self.flush()
        return window.tip_panel.text()

    @staticmethod
    def long_headers(count=7):
        return [f"WP_{number:03d}." + "Hypothetical_protein_" * 4 + "end" for number in range(count)]

    def test_the_report_is_never_cut_off(self):
        window, panel = self.window, self.window.tip_panel
        for width in (1000, window.minimumSizeHint().width()):
            with self.subTest(width=width):
                self.run_check(self.long_headers(), "WP_005", msa_lacks=5)
                window.resize(width, 650)
                self.flush()
                # The splitter keeps no label shorter than its text at its width.
                self.assertGreaterEqual(panel.height(), panel.heightForWidth(panel.width()))
                self.assertGreaterEqual(panel.minimumHeight(), panel.heightForWidth(panel.width()))

    def test_a_tip_gives_the_panel_back_its_ordinary_size(self):
        window, panel = self.window, self.window.tip_panel
        self.run_check(self.long_headers(), "WP_005", msa_lacks=5)
        self.assertGreater(panel.minimumHeight(), 0)
        window.eventFilter(window.labels["NODE_FASTA_FILE"], _left_press())
        self.assertEqual(panel.minimumHeight(), 0)
        self.assertIn("Primary FASTA file", panel.text())

    def test_one_long_header_does_not_widen_the_window(self):
        window = self.window
        before = window.minimumSizeHint().width()
        header = "tr" + "A0A0A0_9BACT" * 40
        report = parse_report(self.run_check(["Short_1", header], header[:30] + "*"))
        self.assertEqual(report.rows[-1].result.count(header), 1)
        self.assertEqual(window.minimumSizeHint().width(), before)
        self.assertLess(window.tip_panel.minimumSizeHint().width(), 600)
        window.resize(before, 650)
        self.flush()
        self.assertEqual(window.width(), before)

    def test_the_table_keeps_its_first_columns_on_a_line_and_wraps_the_rest(self):
        report_html = self.run_check(self.long_headers(), "WP_005")
        wide, narrow = lines_per_cell(report_html, 700), lines_per_cell(report_html, 450)
        for rows in (wide, narrow):
            for name_lines, result_lines, _ in rows[:-1]:
                self.assertEqual((name_lines, result_lines), (1, 1))
            self.assertEqual(rows[-1][0], 1)
        # The reference's sentence, the longest text, takes the lines the width takes away.
        self.assertGreater(narrow[-1][1], wide[-1][1])
        self.assertGreater(narrow[-1][1], 2)


def _left_press():
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(1, 1), QPointF(1, 1),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )


if __name__ == "__main__":
    unittest.main()
