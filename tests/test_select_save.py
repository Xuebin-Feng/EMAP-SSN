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
