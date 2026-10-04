"""`select save` is an action on the current selection, never a trailing mode."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from commands import select  # noqa: E402
from tests.test_viewer_command_portal import Viewer  # noqa: E402
from utilities.Sequence_Utils import load_sanitized_fasta, read_fasta  # noqa: E402


class SelectSaveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.header_dir = directory.name
        patcher = mock.patch.object(select.cfg, "HEADER_LIST_DIR", self.header_dir, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_after_an_expression_fails_cleanly(self):
        # Used to raise UnboundLocalError: 'save' was mapped to a mode that no
        # branch handled.
        for args in (['"one"', 'save', 'picked.txt'], ['"one"', 'SAVE'],
                     ['add', '"one"', 'save', 'picked.txt']):
            with self.subTest(args=args):
                viewer = Viewer(self.header_dir)
                viewer.selected_indices = [2]
                with mock.patch.object(select.Command_Engine, "command_failed") as failed:
                    select.run(viewer, args)
                failed.assert_called_once()
                self.assertIn("'save' must come first", failed.call_args.args[1])
                self.assertEqual(viewer.selected_indices, [2])
                self.assertEqual(os.listdir(self.header_dir), [])

    def test_leading_save_writes_the_current_selection(self):
        viewer = Viewer(self.header_dir)
        viewer.selected_indices = [0, 2]
        with mock.patch.object(select.Command_Engine, "command_failed") as failed:
            select.run(viewer, ['save', 'picked'])
        failed.assert_not_called()
        self.assertEqual(
            Path(self.header_dir, "picked.txt").read_text(encoding="utf-8"), "one\nthree\n"
        )

    def test_save_refuses_a_path_and_writes_nothing(self):
        # os.path.join(HEADER_LIST_DIR, name) used to drop the directory for an
        # absolute name, and "..\" climbed out of it.
        root = Path(self.header_dir)
        header_dir = root / "Header_Lists"
        outside = root / "outside"
        outside.mkdir()
        names = (str(outside / "escaped.txt"), r"..\escaped.txt", "../escaped.fasta",
                 r"sub\picked.txt", "picked.txt:hidden")
        with mock.patch.object(select.cfg, "HEADER_LIST_DIR", str(header_dir), create=True):
            for name in names:
                with self.subTest(name=name):
                    viewer = Viewer(self.header_dir)
                    viewer.selected_indices = [0]
                    with mock.patch.object(select.Command_Engine, "command_failed") as failed:
                        select.run(viewer, ['save', name])
                    failed.assert_called_once()
                    self.assertRegex(failed.call_args.args[1],
                                     "path separators|unsupported characters")
        self.assertEqual(os.listdir(outside), [])
        self.assertEqual(os.listdir(root), ["outside"])

    def test_fasta_save_writes_the_canonical_records_the_viewer_holds(self):
        # Used to re-read NODE_FASTA_FILE with SeqIO and look nodes up by its
        # raw headers, so every record whose header sanitizing changed was
        # reported missing from the source FASTA.
        source = tempfile.TemporaryDirectory()
        self.addCleanup(source.cleanup)
        fasta = Path(source.name, "source.fasta")
        fasta.write_text(
            ">WP_012345678.1 hypothetical protein [Escherichia coli]\nMKVLAAGLL\n"
            ">sp|P69905|HBA_HUMAN Hemoglobin subunit alpha\nMVLSPADKTN\n"
            ">plain_header\nMKTAYIAKQR\n",
            encoding="utf-8",
        )
        headers, sequences, _ = load_sanitized_fasta(str(fasta), report=False)
        self.assertNotIn(headers[0], fasta.read_text(encoding="utf-8"))
        for holds in ("_selected_fasta_records", "sequences_map"):
            with self.subTest(holds=holds):
                viewer = Viewer(self.header_dir)
                viewer.full_headers = list(headers)
                viewer.selected_indices = [0, 1, 2]
                if holds == "_selected_fasta_records":
                    viewer._selected_fasta_records = list(zip(headers, sequences))
                else:
                    viewer.sequences_map = dict(zip(headers, sequences))
                with mock.patch.object(select.cfg, "NODE_FASTA_FILE", str(fasta), create=True), \
                        mock.patch.object(select.Command_Engine, "command_failed") as failed:
                    select.run(viewer, ['save', 'picked.fasta'])
                failed.assert_not_called()
                self.assertEqual(
                    read_fasta(str(Path(self.header_dir, "picked.fasta"))),
                    (list(headers), list(sequences)),
                )

    def test_fasta_save_without_loaded_sequences_fails_without_writing(self):
        viewer = Viewer(self.header_dir)
        viewer.selected_indices = [0]
        missing = str(Path(self.header_dir).parent / "no_such_source.fasta")
        with mock.patch.object(select.cfg, "NODE_FASTA_FILE", missing, create=True), \
                mock.patch.object(select.Command_Engine, "command_failed") as failed:
            select.run(viewer, ['save', 'picked.fasta'])
        failed.assert_called_once()
        self.assertIn("no in-memory sequence set", failed.call_args.args[1])
        self.assertEqual(os.listdir(self.header_dir), [])

    def test_modes_still_work_before_and_after_the_expression(self):
        viewer = Viewer(self.header_dir)
        select.run(viewer, ['"one"'])
        select.run(viewer, ['"three"', 'add'])
        self.assertEqual(sorted(viewer.selected_indices), [0, 2])
        select.run(viewer, ['remove', '"one"'])
        self.assertEqual(viewer.selected_indices, [2])

    def test_help_lists_save_as_a_usage_not_a_mode(self):
        with mock.patch("builtins.print") as printed:
            select.print_help()
        text = printed.call_args.args[0]
        self.assertIn("select save <FILENAME>", text.split("Description:", 1)[0])
        modes = text.split("Modes:", 1)[1].split("Saving:", 1)[0]
        self.assertNotIn("save", modes)


if __name__ == "__main__":
    unittest.main()
