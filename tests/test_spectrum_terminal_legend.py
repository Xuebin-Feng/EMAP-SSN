import io
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import SSN_Config as cfg
from commands import spectrum


class TTYStringIO(io.StringIO):
    def isatty(self):
        return True


def ansi_foreground(rgba):
    red, green, blue = (
        int(round(float(channel) * 255.0)) for channel in rgba[:3]
    )
    return f"\033[38;2;{red};{green};{blue}m"


class SpectrumTerminalLegendTests(unittest.TestCase):
    def make_viewer(self, values):
        values = np.asarray(values, dtype=float)
        node_count = len(values)
        return SimpleNamespace(
            n_nodes=node_count,
            full_headers=[f"node_{index}" for index in range(node_count)],
            cluster_labels=None,
            group_labels=[set() for _ in range(node_count)],
            alignment=None,
            metadata={"Length": {"type": "number", "values": values}},
            current_colors=np.zeros((node_count, 4), dtype=float),
            visible_mask=np.ones(node_count, dtype=bool),
            selected_indices=[],
            console_text=SimpleNamespace(text=""),
            _save_state=mock.Mock(),
            promote_nodes=mock.Mock(),
            update_nodes=mock.Mock(),
            update_console_background=mock.Mock(),
        )

    def run_spectrum(self, viewer, args, output):
        fake_meta = SimpleNamespace(run=mock.Mock())
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            cfg, "HEADER_LIST_DIR", temp_dir, create=True
        ), mock.patch(
            "importlib.import_module", return_value=fake_meta
        ), mock.patch.object(
            sys, "stdout", output
        ):
            spectrum.run(viewer, args)

    def test_terminal_colors_min_and_max_with_colormap_endpoints(self):
        viewer = self.make_viewer([1.0, np.nan, 3.0])
        output = TTYStringIO()

        self.run_spectrum(viewer, ["{Length}", "coolwarm"], output)

        cmap, _ = spectrum.get_colormap("coolwarm")
        self.assertIn(
            f"{ansi_foreground(cmap(0.0))}min: 1.0\033[0m", output.getvalue()
        )
        self.assertIn(
            f"{ansi_foreground(cmap(1.0))}max: 3.0\033[0m", output.getvalue()
        )
        self.assertNotIn("\033[", viewer.console_text.text)
        self.assertIn("(min: 1.0, max: 3.0)", viewer.console_text.text)
        self.assertEqual(output.getvalue().count("Spectrum coloring applied"), 1)

    def test_constant_range_uses_the_applied_midpoint_color_for_both_labels(self):
        viewer = self.make_viewer([2.0, 2.0])
        output = TTYStringIO()

        self.run_spectrum(viewer, ["viridis", "{Length}"], output)

        cmap, _ = spectrum.get_colormap("viridis")
        midpoint = ansi_foreground(cmap(0.5))
        self.assertIn(f"{midpoint}min: 2.0\033[0m", output.getvalue())
        self.assertIn(f"{midpoint}max: 2.0\033[0m", output.getvalue())

    def test_redirected_output_remains_plain_text(self):
        viewer = self.make_viewer([1.0, 3.0])
        output = io.StringIO()

        self.run_spectrum(viewer, ["{Length}"], output)

        self.assertNotIn("\033[", output.getvalue())
        self.assertIn("(min: 1.0, max: 3.0)", output.getvalue())

    def test_invalid_scheme_uses_fallback_colormap_endpoint_colors(self):
        viewer = self.make_viewer([1.0, 3.0])
        output = TTYStringIO()

        self.run_spectrum(viewer, ["not-a-map", "{Length}"], output)

        cmap, _ = spectrum.get_colormap("coolwarm")
        self.assertIn(
            f"{ansi_foreground(cmap(0.0))}min: 1.0\033[0m", output.getvalue()
        )
        self.assertIn(
            f"{ansi_foreground(cmap(1.0))}max: 3.0\033[0m", output.getvalue()
        )
        self.assertIn("using coolwarm", viewer.console_text.text)
        self.assertNotIn("\033[", viewer.console_text.text)

    def test_flexible_expression_property_and_scheme_order(self):
        viewer = self.make_viewer([1.0, 2.0, 3.0])
        output = io.StringIO()

        self.run_spectrum(viewer, ["plasma", '"node_1"', "{Length}"], output)

        promoted_mask = viewer.promote_nodes.call_args.args[0]
        np.testing.assert_array_equal(promoted_mask, [False, True, False])
        viewer._save_state.assert_called_once_with()

    def test_metadata_predicate_is_distinct_from_property_selector(self):
        viewer = self.make_viewer([1.0, 2.0, 3.0])
        output = io.StringIO()

        self.run_spectrum(viewer, ["{Length>1}", "{Length}"], output)

        promoted_mask = viewer.promote_nodes.call_args.args[0]
        np.testing.assert_array_equal(promoted_mask, [False, True, True])

    def test_native_selection_expression_targets_selected_nodes(self):
        viewer = self.make_viewer([1.0, 2.0, 3.0])
        viewer.selected_indices = [0, 2]
        output = io.StringIO()

        self.run_spectrum(viewer, ["$sele$", "{Length}"], output)

        promoted_mask = viewer.promote_nodes.call_args.args[0]
        np.testing.assert_array_equal(promoted_mask, [True, False, True])

    def test_legacy_prefixes_are_rejected_without_mutation(self):
        for legacy in ("prop:Length", "property:Length", "scheme:viridis", "color:plasma"):
            with self.subTest(legacy=legacy):
                viewer = self.make_viewer([1.0, 2.0])
                self.run_spectrum(viewer, [legacy, "{Length}"], io.StringIO())
                self.assertIn("Legacy spectrum prefixes", viewer.console_text.text)
                viewer._save_state.assert_not_called()

    def test_duplicate_or_missing_roles_are_rejected(self):
        cases = (
            (["{Length}", "{Length}"], "exactly one"),
            (["{Length}", "viridis", "plasma"], "at most one color"),
            (["{Length}", '"node_0"', '"node_1"'], "at most one Boolean"),
            (["viridis"], "{PROPERTY_NAME}"),
        )
        for args, message in cases:
            with self.subTest(args=args):
                viewer = self.make_viewer([1.0, 2.0])
                self.run_spectrum(viewer, args, io.StringIO())
                self.assertIn(message, viewer.console_text.text)
                viewer._save_state.assert_not_called()

    def test_text_property_is_rejected(self):
        viewer = self.make_viewer([1.0, 2.0])
        viewer.metadata["Organism"] = {
            "type": "text",
            "values": np.array(["a", "b"], dtype=object),
        }

        self.run_spectrum(viewer, ["{Organism}"], io.StringIO())

        self.assertIn("is not numerical", viewer.console_text.text)
        viewer._save_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
