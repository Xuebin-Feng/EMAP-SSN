"""Output names stay in the folder the command writes to, whoever issues it."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import Metadata_Core  # noqa: E402
from commands import logo, meta as meta_command  # noqa: E402
from utilities.Output_Names import validate_output_basename  # noqa: E402


class OutputBasenameTests(unittest.TestCase):
    def test_plain_names_are_returned_without_surrounding_space(self):
        for name, expected in [
            ("hits", "hits"),
            (" hits.txt ", "hits.txt"),
            ("my traits.csv", "my traits.csv"),
            (".hidden", ".hidden"),
            ("...", "..."),
        ]:
            with self.subTest(name=name):
                self.assertEqual(validate_output_basename(name), expected)

    def test_paths_are_refused(self):
        # os.path.join(folder, name) drops the folder for each of these, and on
        # Windows os.path.isabs misses "C:x" and "\x" altogether.
        for name in [
            r"C:\Users\Public\x.csv",
            r"\x.csv",
            r"\\host\share\x.csv",
            "/tmp/x.csv",
            r"..\x.csv",
            "../x.csv",
            r"sub\x.csv",
            "sub/x.csv",
            ".",
            "..",
        ]:
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "must not include a directory or path separators"
            ):
                validate_output_basename(name)

    def test_drive_and_stream_names_are_refused(self):
        # "C:x" is relative to drive C's current directory; "x.txt:hidden"
        # writes an NTFS alternate data stream inside x.txt.
        for name in ["C:x.csv", "x.txt:hidden", "a<b", "a>b", 'a"b', "a|b",
                     "a?b", "a*b", "a\x00b", "a\tb"]:
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "unsupported characters"
            ):
                validate_output_basename(name)

    def test_empty_names_are_refused(self):
        for name in ("", "   "):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "cannot be empty"
            ):
                validate_output_basename(name)

    def test_logo_keeps_its_extension_rule(self):
        self.assertEqual(logo._normalize_logo_filename("motif"), "motif.svg")
        self.assertEqual(logo._normalize_logo_filename("motif.PNG"), "motif.PNG")
        with self.assertRaisesRegex(ValueError, "path separators"):
            logo._normalize_logo_filename(r"..\outside.svg")


class MetadataDownloadPathTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.meta_dir = directory.name

    def test_requested_names_stay_in_the_metadata_directory(self):
        for name, expected in [
            ("traits", "traits.csv"),
            ("traits.CSV", "traits.CSV"),
            ("traits.xlsx", "traits.xlsx"),
            ("traits.XLSX", "traits.XLSX"),
            ("my traits.csv", "my traits.csv"),
        ]:
            with self.subTest(name=name):
                self.assertEqual(
                    Metadata_Core.metadata_download_path(self.meta_dir, name),
                    os.path.join(self.meta_dir, expected),
                )

    def test_other_extensions_and_paths_are_refused(self):
        for name, message in [
            ("traits.xls", "only be downloaded as .csv or .xlsx, not '.xls'"),
            ("traits.xlsm", "only be downloaded as .csv or .xlsx"),
            ("traits.txt", "only be downloaded as .csv or .xlsx"),
            ("traits.", "only be downloaded as .csv or .xlsx"),
            (r"..\traits.csv", "path separators"),
            ("C:traits.csv", "unsupported characters"),
        ]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                Metadata_Core.metadata_download_path(self.meta_dir, name)

    def test_automatic_names_skip_existing_files(self):
        first = Metadata_Core.metadata_download_path(self.meta_dir)
        self.assertEqual(first, os.path.abspath(os.path.join(self.meta_dir, "metadata.csv")))
        Path(first).write_text("taken", encoding="utf-8")
        self.assertEqual(
            Metadata_Core.metadata_download_path(self.meta_dir, ""),
            os.path.abspath(os.path.join(self.meta_dir, "metadata1.csv")),
        )


class MetaDownloadCommandTests(unittest.TestCase):
    """The desktop `meta download` command, as the console, agent and MCP run it."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.meta_dir = self.root / "Meta_Data"
        self.outside = self.root / "outside"
        self.outside.mkdir()
        patcher = mock.patch.object(
            meta_command.cfg, "METADATA_DIR", str(self.meta_dir), create=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.viewer = SimpleNamespace(
            n_nodes=2,
            full_headers=["node-1", "node-2"],
            metadata={"Score": {"type": "number", "values": np.array([1.0, 2.5])}},
        )

    def download(self, *names):
        engine = meta_command.Command_Engine
        with mock.patch.object(engine, "print_help"), \
                mock.patch.object(engine, "command_artifact"), \
                mock.patch.object(engine, "command_succeeded") as succeeded, \
                mock.patch.object(engine, "command_failed") as failed:
            meta_command.run(self.viewer, ["download", *names])
        return succeeded, failed

    def test_a_path_is_refused_and_nothing_is_written(self):
        # os.path.join(METADATA_DIR, name) used to drop the directory for an
        # absolute name, and "..\" climbed out of it.
        for name in (str(self.outside / "escaped.csv"), r"..\escaped.csv",
                     "../escaped.csv"):
            with self.subTest(name=name):
                succeeded, failed = self.download(name)
                succeeded.assert_not_called()
                failed.assert_called_once()
                self.assertIn("path separators", failed.call_args.args[1])
        self.assertEqual(list(self.outside.iterdir()), [])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["Meta_Data", "outside"])
        self.assertEqual(list(self.meta_dir.iterdir()), [])

    def test_names_with_spaces_and_upper_case_extensions_are_written(self):
        # "book.XLSX" used to fail inside pandas: "No engine for filetype: 'XLSX'".
        for names, written in [(("my", "traits"), "my traits.csv"),
                               (("upper.CSV",), "upper.CSV"),
                               (("book.XLSX",), "book.XLSX")]:
            with self.subTest(names=names):
                succeeded, failed = self.download(*names)
                failed.assert_not_called()
                succeeded.assert_called_once()
                self.assertTrue((self.meta_dir / written).is_file())
        workbook = pd.read_excel(self.meta_dir / "book.XLSX", header=None, engine="openpyxl")
        self.assertEqual(workbook.iloc[2:, 0].tolist(), ["node-1", "node-2"])
        self.assertEqual(workbook.iloc[2:, 1].tolist(), [1, 2.5])

    def test_other_extensions_are_refused_before_writing(self):
        for name in ("traits.xls", "traits.xlsm", "traits.txt"):
            with self.subTest(name=name):
                succeeded, failed = self.download(name)
                succeeded.assert_not_called()
                self.assertIn("only be downloaded as .csv or .xlsx", failed.call_args.args[1])
        self.assertEqual(list(self.meta_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
