"""Regression tests for the original fire-and-forget ESMFold command."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands import esmfold as esmfold_command


class ESMFoldCommandTests(unittest.TestCase):
    def test_no_selection_reports_error_without_launching_worker_or_ui(self):
        viewer = SimpleNamespace(
            selected_indices=[],
            selected_node_idx=None,
            console_text=SimpleNamespace(text=""),
        )

        with (
            mock.patch.object(esmfold_command.esmfold_backend, "register") as register,
            mock.patch.object(
                esmfold_command.esmfold_backend,
                "open_esmfold_ui",
            ) as open_esmfold_ui,
            redirect_stdout(io.StringIO()) as output,
        ):
            esmfold_command.run(viewer, [])

        register.assert_not_called()
        open_esmfold_ui.assert_not_called()
        self.assertIn("Error: No nodes selected.", output.getvalue())
        self.assertEqual(viewer.console_text.text, "Error: No nodes selected.")


if __name__ == "__main__":
    unittest.main()
