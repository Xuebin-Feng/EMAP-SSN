"""Viewer metadata: deleting columns, undo/redo, the meta command, the web page's
column headers, value formatting, and filtered downloads."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import h5py
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import EMAPSSN_Config as cfg
from EMAPSSN_Viewer import MainViewer
from commands import meta as meta_command
from commands import save as save_command
from tests.command_fixtures import one_node_viewer, reported_outcomes
from tests.layout_fixtures import make_provenance
from web_ui import meta_backend
from web_ui.Plugin_Manager import WebPluginRegistry


class FakeDisplay:
    def __init__(self):
        self.hidden = False
        self.messages = []

    def hide(self):
        self.hidden = True

    def show(self, message):
        self.hidden = False
        self.messages.append(message)


def make_viewer():
    viewer = MainViewer.__new__(MainViewer)
    viewer.n_nodes = 2
    viewer.full_headers = ["node-1", "node-2"]
    viewer.visible_mask = np.array([True, True], dtype=bool)
    viewer.selected_indices = []
    viewer.selected_node_idx = 0
    viewer.metadata = {
        "Length": {
            "type": "number",
            "values": np.array([100, 110], dtype=np.int32),
        },
        "Organism": {
            "type": "text",
            "values": np.array(["alpha", "beta"], dtype=object),
        },
        "Host": {
            "type": "text",
            "values": np.array(["plant", "soil"], dtype=object),
        },
    }
    viewer.position_history = []
    viewer.redo_stack = []
    viewer.console_text = SimpleNamespace(text="")
    viewer.hud_displays = {"meta_display": FakeDisplay()}
    viewer.meta_display_prop = None
    viewer.update_nodes = mock.Mock()
    viewer.canvas = SimpleNamespace(update=mock.Mock())
    viewer.broadcast_event = mock.Mock()
    return viewer


class MetadataColumnDeletionTests(unittest.TestCase):
    def test_multiple_case_insensitive_names_are_atomic_and_deduplicated(self):
        viewer = make_viewer()

        deleted = meta_backend.delete_metadata_columns(
            viewer, ["organism", "HOST", "Organism"]
        )

        self.assertEqual(deleted, ["Organism", "Host"])
        self.assertEqual(list(viewer.metadata), ["Length"])
        self.assertEqual(len(viewer.position_history), 1)
        event = viewer.broadcast_event.call_args.args[0]
        self.assertEqual(event["columns"], ["Node ID", "Length"])
        self.assertEqual(event["types"], {"Length": "number"})

    def test_missing_or_protected_name_aborts_without_mutation(self):
        for requested in (["Organism", "Missing"], ["Organism", "Node ID"]):
            with self.subTest(requested=requested):
                viewer = make_viewer()
                original = list(viewer.metadata)

                with self.assertRaises(meta_backend.MetadataColumnDeleteError):
                    meta_backend.delete_metadata_columns(viewer, requested)

                self.assertEqual(list(viewer.metadata), original)
                self.assertEqual(viewer.position_history, [])
                viewer.broadcast_event.assert_not_called()

    def test_all_keyword_is_not_supported(self):
        viewer = make_viewer()
        with self.assertRaisesRegex(
            meta_backend.MetadataColumnDeleteError, "not supported"
        ):
            meta_backend.delete_metadata_columns(viewer, ["all"])

    def test_length_is_a_normal_deletable_metadata_column(self):
        viewer = make_viewer()

        deleted = meta_backend.delete_metadata_columns(viewer, ["length"])

        self.assertEqual(deleted, ["Length"])
        self.assertNotIn("Length", viewer.metadata)

    def test_deleted_length_is_not_regenerated(self):
        viewer = make_viewer()
        viewer.metadata = {}
        viewer.sequences_map = {"node-1": "AAAA", "node-2": "AAAAA"}

        viewer._init_colors()

        self.assertNotIn("Length", viewer.metadata)

    def test_delete_undo_redo_restores_order_values_and_active_hud(self):
        viewer = make_viewer()
        display = viewer.hud_displays["meta_display"]
        viewer.meta_display_prop = "Organism"

        meta_backend.delete_metadata_columns(viewer, ["Organism", "Host"])
        self.assertEqual(list(viewer.metadata), ["Length"])
        self.assertIsNone(viewer.meta_display_prop)
        self.assertTrue(display.hidden)

        self.assertTrue(viewer._do_undo())
        self.assertEqual(list(viewer.metadata), ["Length", "Organism", "Host"])
        np.testing.assert_array_equal(
            viewer.metadata["Organism"]["values"], ["alpha", "beta"]
        )
        self.assertEqual(viewer.meta_display_prop, "Organism")
        self.assertEqual(display.messages[-1], "Organism: alpha")

        self.assertTrue(viewer._do_redo())
        self.assertEqual(list(viewer.metadata), ["Length"])
        self.assertIsNone(viewer.meta_display_prop)
        self.assertTrue(display.hidden)

    def test_cell_edit_uses_compact_shared_history(self):
        viewer = make_viewer()

        self.assertTrue(meta_backend.handle_edit_cell(
            viewer,
            {"row": 1, "column": "Organism", "value": "gamma"},
        ))
        self.assertEqual(viewer.metadata["Organism"]["values"][1], "gamma")
        self.assertEqual(
            viewer.position_history[-1]["_history_kind"], "metadata_cell"
        )

        viewer._do_undo()
        self.assertEqual(viewer.metadata["Organism"]["values"][1], "beta")
        viewer._do_redo()
        self.assertEqual(viewer.metadata["Organism"]["values"][1], "gamma")

    def test_subsequent_save_excludes_deleted_columns(self):
        viewer = make_viewer()
        viewer.pos = np.zeros((2, 2), dtype=np.float32)
        viewer.cache_manifest_id = "test-manifest"
        viewer._cache_provenance = make_provenance(viewer.cache_manifest_id)
        meta_backend.delete_metadata_columns(
            viewer, ["Length", "Organism"], broadcast=False
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            default_path = os.path.join(temp_dir, "version_00.h5")
            with mock.patch.object(
                save_command,
                "resolve_selected_cache",
                return_value=default_path,
            ), mock.patch.object(
                save_command.cache_manifest,
                "read_manifest",
                return_value={"manifest_id": "test-manifest"},
            ), mock.patch.object(
                save_command.cache_manifest, "validate_cache_filename"
            ), mock.patch.object(
                save_command.Command_Engine, "print_help"
            ):
                save_command.run(viewer, ["deleted_columns.h5"])

            saved_path = os.path.join(temp_dir, "deleted_columns.h5")
            with h5py.File(saved_path, "r") as handle:
                self.assertEqual(list(handle["metadata"].keys()), ["Host"])

    def test_web_actions_include_delete_and_server_history(self):
        viewer = make_viewer()
        registry = WebPluginRegistry(viewer)

        meta_backend.register_backend(registry, viewer)

        self.assertIn("delete_columns", registry.actions)
        self.assertIn("metadata_undo", registry.actions)
        self.assertIn("metadata_redo", registry.actions)


class MetadataCliDeletionTests(unittest.TestCase):
    def test_delete_remove_and_clear_dispatch_the_same_multiple_column_helper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for alias in ("delete", "remove", "clear"):
                with self.subTest(alias=alias), mock.patch.object(
                    meta_command.cfg, "METADATA_DIR", temp_dir
                ), mock.patch.object(
                    meta_command, "register"
                ) as register_mock, mock.patch.object(
                    meta_command,
                    "delete_metadata_columns",
                    return_value=["Organism", "Host"],
                ) as delete_mock, mock.patch.object(
                    meta_command.Command_Engine, "print_help"
                ):
                    viewer = SimpleNamespace()

                    meta_command.run(viewer, [alias, "organism", "HOST"])

                    register_mock.assert_not_called()
                    delete_mock.assert_called_once_with(
                        viewer, ["organism", "HOST"], broadcast=False
                    )

    def test_delete_without_names_prints_usage(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            meta_command.cfg, "METADATA_DIR", temp_dir
        ), mock.patch.object(
            meta_command.Command_Engine, "print_help"
        ) as print_help:
            meta_command.run(SimpleNamespace(), ["delete"])

        self.assertIn("Usage: meta delete", str(print_help.call_args.args[1]))


class MetadataWebMarkupTests(unittest.TestCase):
    def test_web_headers_have_confirmed_delete_and_server_undo_actions(self):
        html = (SRC_DIR / "web_ui" / "meta.html").read_text(encoding="utf-8")

        self.assertIn('className = "metadata-delete-column"', html)
        self.assertIn("window.confirm(t('Delete metadata column \"{column}\"?'", html)
        self.assertIn('action: "delete_columns"', html)
        self.assertIn('action: "metadata_undo"', html)
        self.assertIn('action: "metadata_redo"', html)
        self.assertIn('title: t("Sequence Header")', html)
        self.assertNotIn('columnName.toLowerCase() !== "length"', html)
        self.assertIn('event.stopPropagation()', html)
        self.assertIn('.metadata-typed-column .tabulator-header-filter', html)
        self.assertIn('content: var(--number-badge, "NUM")', html)
        self.assertIn('content: var(--text-badge, "TXT")', html)
        self.assertIn('cssClass: `metadata-typed-column metadata-${type', html)
        # Nine rules in meta.html centre their content; this is the header title's.
        self.assertRegex(html, r'\.metadata-column-title \{[^}]*justify-content: center;')
        self.assertIn('className = "metadata-column-name"', html)
        self.assertIn('className = "metadata-column-drag-handle"', html)
        self.assertIn('right: -4px;', html)
        self.assertIn('#spreadsheet-table .tabulator-col-sorter', html)
        self.assertIn('right: 10px !important;', html)


class MetadataValueFormattingTests(unittest.TestCase):
    def test_display_drops_the_decimal_only_from_integral_numbers(self):
        cases = [
            (np.float64(500.0), "500"),
            (np.float64(500.5), "500.5"),
            (np.float64(1234567.0), "1234567"),
            (np.float64(-12.0), "-12"),
            ("alpha", "alpha"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(
                    meta_backend.format_metadata_value(value), expected
                )

    def test_export_keeps_cells_numeric_while_dropping_the_decimal(self):
        integral = meta_backend.export_metadata_value(np.float64(500.0))
        self.assertIsInstance(integral, int)
        self.assertEqual(integral, 500)

        fractional = meta_backend.export_metadata_value(np.float64(500.5))
        self.assertEqual(fractional, 500.5)
        self.assertEqual(meta_backend.export_metadata_value("alpha"), "alpha")

    def test_hud_renders_a_float_backed_length_without_a_decimal(self):
        viewer = make_viewer()
        viewer.metadata["Length"]["values"] = np.array([100.0, 110.0])
        display = viewer.hud_displays["meta_display"]

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            meta_command.cfg, "METADATA_DIR", temp_dir
        ), mock.patch.object(meta_command, "register"), mock.patch.object(
            meta_command.Command_Engine, "print_help"
        ):
            meta_command.run(viewer, ["display", "Length"])

        self.assertEqual(display.messages[-1], "Length: 100")

    def test_download_writes_integral_numbers_without_a_decimal(self):
        viewer = make_viewer()
        viewer.metadata["Length"]["values"] = np.array([100.0, 110.5])

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            meta_backend.Command_Engine, "print_help"
        ), mock.patch.object(
            meta_backend.Command_Engine, "command_artifact"
        ), mock.patch.object(
            meta_backend.Command_Engine, "command_succeeded"
        ):
            target = os.path.join(temp_dir, "metadata.csv")
            self.assertTrue(meta_backend.download_metadata(viewer, target))
            written = Path(target).read_text(encoding="utf-8").splitlines()

        self.assertEqual(written[2].split(",")[1], "100")
        self.assertEqual(written[3].split(",")[1], "110.5")


class MetadataFilteredDownloadTests(unittest.TestCase):
    """download_metadata(expr=...), which the VR meta command uses."""

    def make_viewer(self):
        return one_node_viewer()

    def test_invalid_metadata_export_does_not_create_output_file(self):
        viewer = self.make_viewer()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "metadata.csv")
            with mock.patch.object(cfg, "HEADER_LIST_DIR", temp_dir, create=True):
                result = meta_backend.download_metadata(
                    viewer,
                    output_path,
                    expr="{Missing=1}",
                )

            self.assertFalse(result)
            self.assertFalse(os.path.exists(output_path))


def write_metadata_csv(folder, name, rows, encoding="utf-8"):
    """Write rows (headers, types, then data) as a CSV file and return its path."""
    path = os.path.join(folder, name)
    text = "".join(",".join(row) + "\n" for row in rows)
    with open(path, "w", encoding=encoding, newline="") as handle:
        handle.write(text)
    return path


class MetadataSerializationTests(unittest.TestCase):
    """get_serializable_metadata: the rows the web table is sent."""

    def test_a_column_named_id_cannot_overwrite_the_row_index(self):
        viewer = make_viewer()
        viewer.metadata["id"] = {
            "type": "number",
            "values": np.array([77.0, 88.0]),
        }

        rows = viewer.get_serializable_metadata()

        self.assertEqual([row["id"] for row in rows], [0, 1])
        self.assertEqual([row["Node ID"] for row in rows], ["node-1", "node-2"])
        # Every other column is sent as before.
        self.assertEqual([row["Organism"] for row in rows], ["alpha", "beta"])
        self.assertEqual([row["Length"] for row in rows], [100, 110])

    def test_the_payload_of_ordinary_columns_is_unchanged(self):
        viewer = make_viewer()
        viewer.metadata["ID"] = {"type": "text", "values": np.array(["a", "b"], dtype=object)}

        rows = viewer.get_serializable_metadata()

        self.assertEqual(rows[1], {
            "id": 1, "Node ID": "node-2", "Length": 110,
            "Organism": "beta", "Host": "soil", "ID": "b",
        })
        self.assertEqual(list(rows[1]), ["id", "Node ID", "Length", "Organism", "Host", "ID"])

    def test_infinite_numbers_are_sent_as_text_the_browser_can_parse(self):
        viewer = make_viewer()
        viewer.metadata["Score"] = {
            "type": "number",
            "values": np.array([np.inf, -np.inf]),
        }
        viewer.metadata["Ratio"] = {
            "type": "number",
            "values": np.array([np.nan, 2.5], dtype=np.float32),
        }

        rows = viewer.get_serializable_metadata()

        self.assertEqual([row["Score"] for row in rows], ["inf", "-inf"])
        self.assertEqual([row["Ratio"] for row in rows], ["", 2.5])
        # JSON.parse rejects Infinity and NaN, which json.dumps writes by default.
        json.dumps(rows, allow_nan=False)


class MetadataUploadTests(unittest.TestCase):
    """Metadata_Core.upload_metadata, as the desktop viewer calls it."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.viewer = make_viewer()
        self.viewer._save_state = mock.Mock()
        self.viewer.metadata = {}

    def upload(self, *paths):
        with mock.patch.object(meta_backend.Command_Engine, "command_failed"), \
                mock.patch.object(meta_backend.Command_Engine, "command_succeeded"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            meta_backend.upload_metadata(self.viewer, list(paths))
        return output.getvalue()

    def test_a_failed_upload_takes_no_undo_snapshot(self):
        too_short = write_metadata_csv(self.folder, "short.csv", [["", "Score"], ["", "number"]])
        no_match = write_metadata_csv(
            self.folder, "nomatch.csv", [["", "Score"], ["", "number"], ["other", "1"]]
        )
        missing = os.path.join(self.folder, "missing.csv")

        self.upload(too_short, no_match, missing)

        self.viewer._save_state.assert_not_called()
        self.assertEqual(self.viewer.metadata, {})

    def test_a_successful_upload_takes_one_snapshot_before_changing_metadata(self):
        first = write_metadata_csv(
            self.folder, "a.csv", [["", "Score"], ["", "number"], ["node-1", "1"]]
        )
        second = write_metadata_csv(
            self.folder, "b.csv", [["", "Kind"], ["", "text"], ["node-2", "x"]]
        )
        bad = write_metadata_csv(self.folder, "c.csv", [["", "Score"], ["", "number"]])
        seen = []
        self.viewer._save_state.side_effect = lambda: seen.append(dict(self.viewer.metadata))

        self.upload(bad, first, second)

        self.assertEqual(self.viewer._save_state.call_count, 1)
        self.assertEqual(seen, [{}])
        self.assertEqual(list(self.viewer.metadata), ["Score", "Kind"])

    def test_the_reserved_row_index_name_is_refused(self):
        path = write_metadata_csv(
            self.folder, "id.csv", [["", "id", "Score"], ["", "number", "number"], ["node-1", "5", "1"]]
        )
        with mock.patch.object(meta_backend.Command_Engine, "print_help") as print_help, \
                mock.patch.object(meta_backend.Command_Engine, "command_failed"), \
                contextlib.redirect_stdout(io.StringIO()):
            meta_backend.upload_metadata(self.viewer, [path])

        self.assertEqual(self.viewer.metadata, {})
        self.viewer._save_state.assert_not_called()
        self.assertIn("'id' are reserved for the metadata table's row index", str(print_help.call_args.args[1]))

    def test_other_spellings_of_id_are_still_accepted(self):
        path = write_metadata_csv(
            self.folder, "ID.csv", [["", "ID", "Id"], ["", "text", "text"], ["node-1", "a", "b"]]
        )

        self.upload(path)

        self.assertEqual(list(self.viewer.metadata), ["ID", "Id"])

    def test_unknown_types_and_duplicate_names_are_warned_about(self):
        path = write_metadata_csv(
            self.folder, "warn.csv",
            [["", "Score", "Count", "Score", "Name", "Plain"],
             ["", "numeric", "int", "number", "string", ""],
             ["node-1", "1", "2", "3", "x", "y"]],
        )

        output = self.upload(path)

        self.assertIn("Property 'Score' in warn.csv has the unrecognized type 'numeric'", output)
        self.assertIn("Property 'Count' in warn.csv has the unrecognized type 'int'", output)
        self.assertIn("Property 'Score' appears in more than one column of warn.csv", output)
        # The recognized and blank types are not warned about.
        self.assertNotIn("Property 'Name'", output)
        self.assertNotIn("Property 'Plain'", output)
        # Parsing is as before: an unknown type is text, the last Score column wins.
        self.assertEqual(self.viewer.metadata["Count"]["type"], "text")
        self.assertEqual(self.viewer.metadata["Score"]["type"], "number")
        self.assertEqual(self.viewer.metadata["Score"]["values"][0], 3.0)

    def test_a_windows_code_page_csv_is_read(self):
        path = write_metadata_csv(
            self.folder, "excel.csv",
            [["", "Organism"], ["", "text"], ["node-1", "Café µm"]],
            encoding="cp1252",
        )
        with self.assertRaises(UnicodeDecodeError):
            Path(path).read_text(encoding="utf-8")

        self.upload(path)

        self.assertEqual(self.viewer.metadata["Organism"]["values"][0], "Café µm")

    def test_a_utf8_csv_is_still_read_as_utf8(self):
        path = write_metadata_csv(
            self.folder, "utf8.csv",
            [["", "Organism"], ["", "text"], ["node-1", "Café µm"]],
        )

        output = self.upload(path)

        self.assertEqual(self.viewer.metadata["Organism"]["values"][0], "Café µm")
        self.assertNotIn("Windows-1252", output)


class MetadataNaWordsTests(unittest.TestCase):
    """Words pandas would read as missing stay text in a text column.

    "NA", "N/A", "None", "null" and "NaN" can be real values there (NA for
    North America or Namibia). A number column still reads them as missing,
    and an empty cell is blank in both.
    """

    WORDS = ["NA", "N/A", "None", "null", "NaN"]

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.viewer = self.make_viewer()

    @staticmethod
    def make_viewer():
        viewer = make_viewer()
        viewer.n_nodes = 8
        viewer.full_headers = [f"n{i}" for i in range(1, 9)]
        viewer.visible_mask = np.ones(8, dtype=bool)
        viewer._save_state = mock.Mock()
        viewer.metadata = {}
        return viewer

    def sheet_rows(self, last_region="Peru"):
        """Header and type rows, then n1-n5 holding each word in every column,
        n6 empty, n7 whitespace only, and n8 ordinary values. The third
        column's property is literally named NA."""
        rows = [["", "Region", "Score", "NA"], ["", "text", "number", "text"]]
        rows += [[f"n{index}", word, word, word] for index, word in enumerate(self.WORDS, 1)]
        rows.append(["n6", "", "", ""])
        rows.append(["n7", "  ", "  ", "  "])
        rows.append(["n8", last_region, "4.5", "x"])
        return rows

    def upload(self, viewer, *paths):
        with mock.patch.object(meta_backend.Command_Engine, "command_failed"), \
                mock.patch.object(meta_backend.Command_Engine, "command_succeeded"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            meta_backend.upload_metadata(viewer, list(paths))
        return output.getvalue()

    def assert_words_kept_in_text_and_missing_in_numbers(self, viewer, last_region="Peru"):
        self.assertEqual(list(viewer.metadata), ["Region", "Score", "NA"])
        self.assertEqual(
            [entry["type"] for entry in viewer.metadata.values()], ["text", "number", "text"]
        )
        # Text: each word exactly as written; an empty or whitespace-only
        # cell is the empty string.
        self.assertEqual(
            list(viewer.metadata["Region"]["values"]), self.WORDS + ["", "", last_region]
        )
        self.assertEqual(list(viewer.metadata["NA"]["values"]), self.WORDS + ["", "", "x"])
        # Number: the words and the empty cells are all missing.
        score = viewer.metadata["Score"]["values"]
        self.assertTrue(np.isnan(score[:7]).all())
        self.assertEqual(score[7], 4.5)

    def test_a_csv_keeps_the_words_in_text_columns_and_blanks_number_columns(self):
        path = write_metadata_csv(self.folder, "words.csv", self.sheet_rows())

        self.upload(self.viewer, path)

        self.assert_words_kept_in_text_and_missing_in_numbers(self.viewer)

    def test_a_windows_code_page_csv_keeps_the_words_too(self):
        path = write_metadata_csv(
            self.folder, "words_cp1252.csv", self.sheet_rows(last_region="Café"),
            encoding="cp1252",
        )
        with self.assertRaises(UnicodeDecodeError):
            Path(path).read_text(encoding="utf-8")

        output = self.upload(self.viewer, path)

        self.assertIn("Windows-1252", output)
        self.assert_words_kept_in_text_and_missing_in_numbers(self.viewer, last_region="Café")

    def test_an_xlsx_keeps_the_words_in_text_columns_and_blanks_number_columns(self):
        import openpyxl

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        for row in self.sheet_rows():
            sheet.append([None if cell == "" else cell for cell in row])
        # Written as a number cell, as Excel stores 4.5, not as text.
        sheet["C10"] = 4.5
        path = os.path.join(self.folder, "words.xlsx")
        workbook.save(path)

        self.upload(self.viewer, path)

        self.assert_words_kept_in_text_and_missing_in_numbers(self.viewer)

    def test_an_excel_error_cell_is_still_blank(self):
        import openpyxl

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["", "Region", "Score"])
        sheet.append(["", "text", "number"])
        # A formula error such as #N/A is not a word the user typed.
        sheet.append(["n1", "#N/A", "#N/A"])
        sheet.append(["n2", "ok", 2])
        path = os.path.join(self.folder, "errors.xlsx")
        workbook.save(path)

        self.upload(self.viewer, path)

        self.assertEqual(list(self.viewer.metadata["Region"]["values"][:2]), ["", "ok"])
        self.assertTrue(np.isnan(self.viewer.metadata["Score"]["values"][0]))
        self.assertEqual(self.viewer.metadata["Score"]["values"][1], 2.0)

    def test_a_number_column_that_gets_a_word_keeps_its_earlier_value(self):
        first = write_metadata_csv(
            self.folder, "first.csv", [["", "Score"], ["", "number"], ["n1", "3"]]
        )
        second = write_metadata_csv(
            self.folder, "second.csv",
            [["", "Score"], ["", "number"], ["n1", "NA"], ["n2", "None"], ["n3", "nan"]],
        )

        self.upload(self.viewer, first)
        self.upload(self.viewer, second)

        score = self.viewer.metadata["Score"]["values"]
        self.assertEqual(score[0], 3.0)
        self.assertTrue(np.isnan(score[1:3]).all())

    def test_a_sequence_header_of_na_matches_its_node_and_an_empty_one_does_not(self):
        self.viewer.full_headers[0] = "NA"
        path = write_metadata_csv(
            self.folder, "header.csv",
            [["", "Region"], ["", "text"], ["NA", "x"], ["", "y"], ["  ", "z"]],
        )

        output = self.upload(self.viewer, path)

        self.assertEqual(self.viewer.metadata["Region"]["values"][0], "x")
        self.assertEqual(list(self.viewer.metadata["Region"]["values"][1:]), [""] * 7)
        self.assertIn("Matched 1 unique node. Ignored 2 rows.", " ".join(output.split()))

    def test_the_words_reach_the_web_table_and_blanks_are_empty(self):
        self.upload(self.viewer, write_metadata_csv(self.folder, "words.csv", self.sheet_rows()))

        rows = self.viewer.get_serializable_metadata()

        self.assertEqual([row["Region"] for row in rows[:5]], self.WORDS)
        self.assertEqual([row["NA"] for row in rows[:5]], self.WORDS)
        self.assertEqual(rows[5]["Region"], "")
        self.assertEqual(rows[5]["Score"], "")
        self.assertEqual(rows[0]["Score"], "")
        self.assertEqual(rows[7]["Score"], 4.5)

    def test_a_download_and_upload_round_trip_keeps_the_words(self):
        self.upload(self.viewer, write_metadata_csv(self.folder, "words.csv", self.sheet_rows()))
        target = os.path.join(self.folder, "exported.csv")
        with mock.patch.object(meta_backend.Command_Engine, "print_help"), \
                mock.patch.object(meta_backend.Command_Engine, "command_artifact"), \
                mock.patch.object(meta_backend.Command_Engine, "command_succeeded"):
            self.assertTrue(meta_backend.download_metadata(self.viewer, target))
        fresh = self.make_viewer()

        self.upload(fresh, target)

        self.assert_words_kept_in_text_and_missing_in_numbers(fresh)


class MetadataDownloadFailureTests(unittest.TestCase):
    def test_a_failed_export_keeps_the_earlier_file_and_leaves_no_partial_one(self):
        for name in ("metadata.xlsx", "metadata.csv"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                viewer = make_viewer()
                target = os.path.join(folder, name)
                with mock.patch.object(meta_backend.Command_Engine, "print_help"), \
                        mock.patch.object(meta_backend.Command_Engine, "command_artifact"), \
                        mock.patch.object(meta_backend.Command_Engine, "command_succeeded"), \
                        mock.patch.object(meta_backend.Command_Engine, "command_failed") as failed:
                    self.assertTrue(meta_backend.download_metadata(viewer, target))
                    with open(target, "rb") as handle:
                        earlier = handle.read()

                    if name.endswith(".csv"):
                        with mock.patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
                            self.assertFalse(meta_backend.download_metadata(viewer, target))
                    else:
                        # A control character is refused by the xlsx writer.
                        viewer.metadata["Organism"]["values"][1] = "bad\x01value"
                        self.assertFalse(meta_backend.download_metadata(viewer, target))

                failed.assert_called_once()
                with open(target, "rb") as handle:
                    self.assertEqual(handle.read(), earlier)
                self.assertEqual(os.listdir(folder), [name])

    def test_a_partly_written_temporary_file_is_removed(self):
        viewer = make_viewer()
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "metadata.csv")

            def write_then_fail(self_df, path, **kwargs):
                Path(path).write_text("partial", encoding="utf-8")
                raise OSError("disk full")

            with mock.patch.object(meta_backend.Command_Engine, "print_help"), \
                    mock.patch.object(meta_backend.Command_Engine, "command_failed"), \
                    mock.patch.object(pd.DataFrame, "to_csv", write_then_fail):
                self.assertFalse(meta_backend.download_metadata(viewer, target))

            self.assertEqual(os.listdir(folder), [])


class MetaCommandArgumentTests(unittest.TestCase):
    """`meta` command parsing: upload paths, show, off, and the metadata folder."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.meta_dir = os.path.join(self.folder, "Meta_Data")
        patch = mock.patch.object(meta_command.cfg, "METADATA_DIR", self.meta_dir)
        patch.start()
        self.addCleanup(patch.stop)

    def run_meta(self, viewer, args):
        with mock.patch.object(meta_command, "register") as register, \
                mock.patch.object(meta_command, "upload_metadata") as upload, \
                mock.patch.object(meta_command.Command_Engine, "print_help") as print_help, \
                reported_outcomes() as (succeeded, failed), \
                contextlib.redirect_stdout(io.StringIO()):
            meta_command.run(viewer, args)
        return SimpleNamespace(
            register=register, upload=upload, print_help=print_help,
            succeeded=succeeded, failed=failed,
        )

    def test_a_path_with_spaces_is_one_file(self):
        spaced = os.path.join(self.folder, "My Data.csv")
        Path(spaced).write_text("x", encoding="utf-8")
        words = spaced.split(" ")

        for args in (["upload", *words], words,
                     ["upload", '"' + words[0], *words[1:-1], words[-1] + '"'],
                     ["import", "'" + words[0], *words[1:-1], words[-1] + "'"]):
            with self.subTest(args=args):
                result = self.run_meta(make_viewer(), args)

                result.failed.assert_not_called()
                result.upload.assert_called_once()
                self.assertEqual(result.upload.call_args.args[1], [os.path.abspath(spaced)])

    def test_quotes_around_one_path_are_removed(self):
        plain = os.path.join(self.folder, "plain.csv")
        Path(plain).write_text("x", encoding="utf-8")

        result = self.run_meta(make_viewer(), ["upload", f'"{plain}"'])

        self.assertEqual(result.upload.call_args.args[1], [os.path.abspath(plain)])

    def test_separate_files_are_still_uploaded_together(self):
        first, second = (os.path.join(self.folder, name) for name in ("a.csv", "b.csv"))
        for path in (first, second):
            Path(path).write_text("x", encoding="utf-8")

        result = self.run_meta(make_viewer(), ["upload", first, second])

        self.assertEqual(
            result.upload.call_args.args[1], [os.path.abspath(first), os.path.abspath(second)]
        )

    def test_show_prefers_an_exact_name_over_a_case_insensitive_one(self):
        viewer = make_viewer()
        viewer.metadata = {
            "Organism": {"type": "text", "values": np.array(["a", "b"], dtype=object)},
            "organism": {"type": "text", "values": np.array(["c", "d"], dtype=object)},
        }
        display = viewer.hud_displays["meta_display"]

        for requested, expected in (("organism", "organism"), ("Organism", "Organism"),
                                    ("ORGANISM", "Organism")):
            with self.subTest(requested=requested):
                result = self.run_meta(viewer, ["show", requested])

                result.failed.assert_not_called()
                self.assertEqual(viewer.meta_display_prop, expected)
                self.assertTrue(display.messages[-1].startswith(f"{expected}: "))

    def test_show_of_an_unknown_property_fails_and_lists_the_properties(self):
        viewer = make_viewer()
        result = self.run_meta(viewer, ["show", "NoSuchColumn"])

        self.assertIsNone(viewer.meta_display_prop)
        self.assertEqual(viewer.hud_displays["meta_display"].messages, [])
        result.succeeded.assert_not_called()
        result.failed.assert_called_once()
        message = result.failed.call_args.args[1]
        self.assertIn("Property 'NoSuchColumn' not found", message)
        self.assertIn("Available properties: Length, Organism, Host.", message)
        self.assertEqual(str(result.print_help.call_args.args[1]), message)

    def test_show_with_no_metadata_fails_and_says_none_is_available(self):
        viewer = make_viewer()
        viewer.metadata = {}

        result = self.run_meta(viewer, ["show", "Length"])

        self.assertIsNone(viewer.meta_display_prop)
        self.assertIn("Available properties: none.", result.failed.call_args.args[1])

    def test_off_and_deactivate_say_they_are_not_metadata_options(self):
        for word in ("off", "deactivate", "OFF"):
            with self.subTest(word=word):
                result = self.run_meta(make_viewer(), [word])

                result.register.assert_not_called()
                result.upload.assert_not_called()
                result.succeeded.assert_not_called()
                message = result.failed.call_args.args[1]
                self.assertIn(f"'{word}' is not a metadata option", message)
                self.assertIn("meta show clear", message)
                self.assertNotIn("not found", message)

    def test_a_metadata_file_named_off_is_still_uploaded(self):
        os.makedirs(self.meta_dir)
        Path(self.meta_dir, "off.csv").write_text("x", encoding="utf-8")

        result = self.run_meta(make_viewer(), ["off"])

        result.failed.assert_not_called()
        self.assertEqual(
            result.upload.call_args.args[1], [os.path.abspath(os.path.join(self.meta_dir, "off.csv"))]
        )

    def test_the_metadata_folder_is_created_only_by_a_download(self):
        for args in (["help"], ["off"], ["show", "Length"], ["delete"], ["upload", "nothing"]):
            with self.subTest(args=args):
                self.run_meta(make_viewer(), args)
                self.assertFalse(os.path.exists(self.meta_dir))

        viewer = one_node_viewer()
        with mock.patch.object(meta_command, "download_metadata") as download:
            self.run_meta(viewer, ["download", "table"])
        self.assertTrue(os.path.isdir(self.meta_dir))
        download.assert_called_once()


class MetadataHudRefreshTests(unittest.TestCase):
    def make_viewer_with_hud(self):
        viewer = make_viewer()
        display = viewer.hud_displays["meta_display"]
        display.visible = True
        display.on_node_clicked = mock.Mock()
        viewer.meta_display_prop = "Organism"
        viewer.selected_node_idx = 1
        return viewer, display

    def test_editing_the_shown_cell_redraws_the_hud(self):
        viewer, display = self.make_viewer_with_hud()

        self.assertTrue(meta_backend.handle_edit_cell(
            viewer, {"row": 1, "column": "Organism", "value": "gamma"}
        ))

        display.on_node_clicked.assert_called_once_with(1)

    def test_editing_another_cell_leaves_the_hud_alone(self):
        viewer, display = self.make_viewer_with_hud()

        meta_backend.handle_edit_cell(viewer, {"row": 0, "column": "Organism", "value": "gamma"})
        meta_backend.handle_edit_cell(viewer, {"row": 1, "column": "Host", "value": "air"})
        display.visible = False
        meta_backend.handle_edit_cell(viewer, {"row": 1, "column": "Organism", "value": "delta"})

        display.on_node_clicked.assert_not_called()

    def test_web_undo_and_redo_redraw_the_visible_hud(self):
        viewer, display = self.make_viewer_with_hud()
        meta_backend.handle_edit_cell(viewer, {"row": 1, "column": "Organism", "value": "gamma"})
        display.on_node_clicked.reset_mock()

        self.assertTrue(meta_backend.handle_metadata_undo(viewer, {}))
        display.on_node_clicked.assert_called_once_with(1)
        display.on_node_clicked.reset_mock()
        self.assertTrue(meta_backend.handle_metadata_redo(viewer, {}))
        display.on_node_clicked.assert_called_once_with(1)
        display.on_node_clicked.reset_mock()

        self.assertFalse(meta_backend.handle_metadata_redo(viewer, {}))
        display.on_node_clicked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
