import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import openpyxl
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager
from commands import label


class AlignmentStub:
    def __init__(self):
        self.aln = MultipleSeqAlignment(
            [
                SeqRecord(Seq("A"), id="node0"),
                SeqRecord(Seq("A"), id="node1"),
                SeqRecord(Seq("C"), id="node2"),
            ]
        )
        self.col_to_label = {0: "1"}
        self.label_to_col = {"1": 0}
        self.has_reference = True
        self.resolved_ref_full = "node0"

    def calculate_frequencies(self, mapping, exclude=None, aln=None):
        return Alignment_Manager.calculate_frequencies(
            aln if aln is not None else self.aln,
            mapping,
            exclude or [],
        )


def find_row(worksheet, first_cell_value):
    for row_idx in range(1, worksheet.max_row + 1):
        if worksheet.cell(row=row_idx, column=1).value == first_cell_value:
            return row_idx
    raise AssertionError(f"Could not find row {first_cell_value!r}")


class LabelWorkbookPercentTests(unittest.TestCase):
    def run_label(self, directory, args):
        viewer = SimpleNamespace(
            alignment=AlignmentStub(),
            active_reference="node0",
            full_headers=["node0", "node1", "node2", "unaligned"],
            n_nodes=4,
            cluster_labels=np.array([0, 0, 1, -1]),
            group_labels=[{"GroupA"}, set(), set(), set()],
            console_text=SimpleNamespace(text=""),
            last_cluster_params=None,
        )
        metadata = SimpleNamespace(model_name="test-model", network_type="cosine")

        with mock.patch.object(label.cfg, "CLUSTER_LABEL_DIR", directory), \
                mock.patch.object(label.cfg, "NODE_FASTA_FILE", "nodes.fasta"), \
                mock.patch.object(label.cfg, "INPUT_HDF5", "network.h5"), \
                mock.patch.object(label.cfg, "MSA_FILE", "alignment.fasta"), \
                mock.patch.object(label.cache_manifest, "validate_network_schema", return_value=metadata), \
                mock.patch.object(label.Command_Engine, "get_alignment_mapping", return_value=(np.array([0, 1, 2, -1]), np.array([0, 1, 2]))), \
                mock.patch.object(label.cluster_cmd, "get_cluster_color_map", return_value={0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0)}), \
                mock.patch.object(label.utils, "open_in_file_manager"):
            label.run(viewer, args)
        return viewer

    def test_percent_column_uses_total_network_nodes_in_both_sheets(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_label(directory, [])

            output_path = next(Path(directory).glob("Label_Output_*.xlsx"))
            workbook = openpyxl.load_workbook(output_path)

            for sheet_name in ("Subset Stats", "Occupancy Stats"):
                worksheet = workbook[sheet_name]
                header_row = find_row(worksheet, "Subset Name")
                cluster_row = find_row(worksheet, "Cluster 0")
                group_row = find_row(worksheet, "Group GroupA")
                global_row = find_row(worksheet, "Global Stats")

                self.assertEqual(worksheet.freeze_panes, "C1")
                self.assertEqual(worksheet.cell(header_row, 2).value, "Percent")
                self.assertEqual(worksheet.cell(header_row, 3).value, "Count")
                self.assertEqual(worksheet.cell(header_row, 4).value, "Hex Color")
                self.assertEqual(worksheet.cell(header_row, 10).value, "#1")
                self.assertEqual(worksheet.cell(cluster_row, 2).value, 0.5)
                self.assertEqual(worksheet.cell(group_row, 2).value, 0.25)
                self.assertEqual(worksheet.cell(global_row, 2).value, 0.75)
                self.assertEqual(worksheet.cell(cluster_row, 2).number_format, "0.00%")
                self.assertEqual(worksheet.column_dimensions["B"].width, 10.0)
                self.assertEqual(worksheet.cell(cluster_row, 4).fill.fill_type, "solid")
                self.assertEqual(worksheet.cell(cluster_row, 10).fill.fill_type, "solid")

    def test_custom_filename_is_used_and_xlsx_is_appended(self):
        with tempfile.TemporaryDirectory() as directory:
            viewer = self.run_label(
                directory,
                ["0.4", "0.9", "custom_label_report"],
            )

            output_path = Path(directory, "custom_label_report.xlsx")
            self.assertTrue(output_path.is_file())
            self.assertIn(str(output_path), viewer.console_text.text)

            workbook = openpyxl.load_workbook(output_path)
            self.assertEqual(workbook["Meta Data"]["B11"].value, (
                "gmax_outside=40%, cmin=90%, "
                "global_conservation_threshold=97% (fixed), target=clusters"
            ))
            for sheet_name in ("Subset Stats", "Occupancy Stats"):
                worksheet = workbook[sheet_name]
                self.assertIsNotNone(find_row(worksheet, "Global Conserved (>97%)"))

    def test_bare_filename_is_allowed_after_keyword_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_label(directory, ["cmin", "90%", "keyword_report"])

            self.assertTrue(Path(directory, "keyword_report.xlsx").is_file())

    def test_numeric_filename_requires_and_accepts_xlsx_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_label(directory, ["0.4.xlsx"])

            self.assertTrue(Path(directory, "0.4.xlsx").is_file())

    def test_filename_must_be_final_and_old_keyword_form_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            viewer = self.run_label(directory, ["report", "cmin", "90%"])
            self.assertEqual(
                viewer.console_text.text,
                "Error: A custom output filename must be the final argument.",
            )
            self.assertEqual(list(Path(directory).glob("*.xlsx")), [])

        with tempfile.TemporaryDirectory() as directory:
            viewer = self.run_label(directory, ["filename", "old_style"])
            self.assertEqual(
                viewer.console_text.text,
                "Error: A custom output filename must be the final argument.",
            )
            self.assertEqual(list(Path(directory).glob("*.xlsx")), [])

    def test_gmin_is_not_user_configurable(self):
        with tempfile.TemporaryDirectory() as directory:
            viewer = self.run_label(directory, ["gmin", "90%"])

            self.assertEqual(
                viewer.console_text.text,
                "Error: gmin is fixed at 97% and cannot be set by the label command.",
            )
            self.assertEqual(list(Path(directory).glob("*.xlsx")), [])

    def test_filename_cannot_escape_output_directory(self):
        with self.assertRaisesRegex(ValueError, "path separators"):
            label._normalize_output_filename("../outside")


if __name__ == "__main__":
    unittest.main()
