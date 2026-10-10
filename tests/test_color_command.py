"""The color command (commands/color.py): residue selections versus Matplotlib
cycle names, scales and shapes, and batches applied all-or-nothing."""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import matplotlib.colors as mcolors
import numpy as np

from tests.command_fixtures import add_alignment, one_node_viewer, reported_outcomes
from tests.sparse_alignment import sparse_alignment
from tests.viewer_fixtures import PortalFixture, Viewer
from commands import color
from commands import color as color_command  # the name ColorBatchTests use
import EMAPSSN_Config as cfg


class CysteineColorTests(unittest.TestCase):
    def viewer(self):
        return add_alignment(Viewer('.'))

    def test_visible_matches_ignore_selection_and_preserve_other_syntax(self):
        for expression in ['C0', 'C1', 'C53', 'C01', 'C1000', 'c53', '(C53)', 'C53.1', 'C(-1)']:
            for selection in [[], [1]]:
                with self.subTest(expression=expression, selection=selection):
                    viewer = self.viewer()
                    viewer.selected_indices = selection
                    color.run(viewer, [expression, 'red'])
                    np.testing.assert_array_equal(viewer.current_colors[0], mcolors.to_rgba('red'))
                    np.testing.assert_array_equal(viewer.current_colors[1:], np.ones((2, 4)))
                    self.assertEqual(viewer.saved, 1)

    def test_missing_position_aborts_entire_batch(self):
        viewer = self.viewer()
        viewer.selected_indices = [1]
        color.run(viewer, ['C53', 'red', 'C999', 'blue'])
        np.testing.assert_array_equal(viewer.current_colors, np.ones((3, 4)))
        self.assertEqual(viewer.saved, 0)

    def test_zero_matches_does_not_fall_back_to_selection(self):
        viewer = self.viewer()
        viewer.alignment.aln = sparse_alignment((f'row{index}', 'K' * 7) for index in range(3))
        viewer.selected_indices = [1]
        color.run(viewer, ['C53', 'red'])
        np.testing.assert_array_equal(viewer.current_colors, np.ones((3, 4)))
        self.assertEqual(viewer.saved, 0)

    def test_properties_still_target_selection(self):
        for value in ['red', '#123abc', 'c']:
            viewer = self.viewer()
            viewer.selected_indices = [1]
            color.run(viewer, [value, '0x', 'square'])
            np.testing.assert_array_equal(viewer.current_colors[1], mcolors.to_rgba(value))
            self.assertEqual(viewer.current_sizes[1], 0)
            self.assertEqual(viewer.current_shapes[1], 'square')

    def test_suffix_scales_and_bare_x_shape(self):
        for token, factor in [('2x', 2), ('0.5x', .5), ('0x', 0)]:
            for expression in [[], ['C53']]:
                viewer = self.viewer()
                viewer.selected_indices = [0]
                color.run(viewer, expression + ['red', token, 'x'])
                self.assertEqual(viewer.current_sizes[0], cfg.NODE_SIZE * factor)
                np.testing.assert_array_equal(viewer.current_sizes[1:], [1, 1])
                self.assertEqual(viewer.current_shapes[0], 'x')

    def test_invalid_scales_are_rejected_before_any_change(self):
        # float() accepts all of these; none is a usable node size.
        for token in ['-2x', '-0.5x', 'nanx', 'infx', '-infx', '1e400x']:
            with self.subTest(token=token):
                viewer = self.viewer()
                viewer.selected_indices = [0]
                with mock.patch.object(color.Command_Engine, 'command_failed') as failed:
                    color.run(viewer, ['red', token, 'C53', 'blue', '2x'])
                failed.assert_called_once()
                self.assertIn(f"Invalid scale '{token}'", str(failed.call_args.args[1]))
                np.testing.assert_array_equal(viewer.current_sizes, np.ones(3))
                np.testing.assert_array_equal(viewer.current_colors, np.ones((3, 4)))
                self.assertEqual(viewer.saved, 0)

    def test_old_prefix_is_only_a_residue_selection(self):
        for token in ['x2', 'X2']:
            for available in [False, True]:
                viewer = self.viewer()
                viewer.selected_indices = [1]
                if available:
                    viewer.alignment.label_to_col['2'] = 0
                    viewer.alignment.aln = sparse_alignment(
                        [('row0', 'X' * 7), ('row1', 'K' * 7), ('row2', 'C' * 7)]
                    )
                color.run(viewer, [token, 'red'])
                np.testing.assert_array_equal(viewer.current_sizes, np.ones(3))
                np.testing.assert_array_equal(viewer.current_colors[1:], np.ones((2, 4)))
                expected = mcolors.to_rgba('red') if available else np.ones(4)
                np.testing.assert_array_equal(viewer.current_colors[0], expected)


class ColorBatchTests(unittest.TestCase):
    def make_viewer(self):
        return one_node_viewer()

    def test_color_does_not_apply_earlier_pair_when_later_pair_is_invalid(self):
        viewer = self.make_viewer()
        original_colors = viewer.current_colors.copy()
        original_sizes = viewer.current_sizes.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(cfg, "HEADER_LIST_DIR", temp_dir, create=True):
                color_command.run(
                    viewer,
                    ['"node"', "red", "#cluster_99#", "blue"],
                )

        np.testing.assert_array_equal(viewer.current_colors, original_colors)
        np.testing.assert_array_equal(viewer.current_sizes, original_sizes)
        self.assertFalse(hasattr(viewer, "current_shapes"))
        viewer._save_state.assert_not_called()
        viewer.promote_nodes.assert_not_called()
        viewer.update_nodes.assert_not_called()

    def test_color_zero_scale_is_applied(self):
        viewer = self.make_viewer()

        color_command.run(viewer, ['"node"', "0x"])

        np.testing.assert_array_equal(viewer.current_sizes, np.array([0.0]))
        viewer._save_state.assert_called_once_with()
        viewer.update_nodes.assert_called_once_with()


class ColorScaleLimitTests(unittest.TestCase):
    def make_viewer(self):
        viewer = one_node_viewer()
        # The Viewer keeps sizes as float32, which overflows before float64 does.
        viewer.current_sizes = np.array([5.0], dtype=np.float32)
        return viewer

    def run_color(self, viewer, arguments):
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
            color_command.run(viewer, arguments)
        return succeeded, failed

    def test_a_scale_whose_size_overflows_float32_is_rejected_before_any_change(self):
        for token in ("1e38x", "3.5e37x", "1e300x"):
            with self.subTest(token=token):
                viewer = self.make_viewer()
                original_colors = viewer.current_colors.copy()
                succeeded, failed = self.run_color(viewer, ['"node"', "red", token])

                message = f"Error: Scale '{token}' is too large to make a valid node size."
                failed.assert_called_once_with(viewer, message)
                succeeded.assert_not_called()
                self.assertEqual(viewer.console_text.text, message)
                np.testing.assert_array_equal(viewer.current_sizes, [5.0])
                np.testing.assert_array_equal(viewer.current_colors, original_colors)
                viewer._save_state.assert_not_called()

    def test_the_largest_scale_that_fits_is_applied(self):
        viewer = self.make_viewer()
        succeeded, failed = self.run_color(viewer, ['"node"', "3e37x"])

        failed.assert_not_called()
        self.assertTrue(np.isfinite(viewer.current_sizes[0]))
        self.assertEqual(viewer.current_sizes[0], np.float32(cfg.NODE_SIZE * 3e37))

    def test_a_negative_zero_scale_is_the_scale_zero(self):
        for token in ("-0x", "-0.0x", "0x"):
            with self.subTest(token=token):
                viewer = self.make_viewer()
                succeeded, failed = self.run_color(viewer, ['"node"', token])

                failed.assert_not_called()
                self.assertEqual(viewer.current_sizes[0], 0.0)
                self.assertFalse(np.signbit(viewer.current_sizes[0]))
                self.assertEqual(
                    succeeded.call_args.args[1], "Applied: 1 node (0.0x)"
                )


class ColorSkippedExpressionTests(unittest.TestCase):
    def make_viewer(self):
        return one_node_viewer()

    def run_color(self, viewer, arguments):
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
            color_command.run(viewer, arguments)
        return succeeded, failed

    def test_a_trailing_expression_without_properties_is_reported(self):
        viewer = self.make_viewer()
        succeeded, failed = self.run_color(
            viewer, ['"node"', "red", "#cluster_1#"]
        )

        failed.assert_not_called()
        np.testing.assert_array_equal(viewer.current_colors[0], mcolors.to_rgba("red"))
        message = (
            "Applied: 1 node (red) "
            "Skipped 1 expression with no color, scale or shape: #cluster_1#."
        )
        succeeded.assert_called_once_with(viewer, message)
        self.assertEqual(viewer.console_text.text, message)

    def test_every_skipped_expression_is_listed(self):
        viewer = self.make_viewer()
        succeeded, failed = self.run_color(
            viewer, ["#noise#", '"node"', "blue", "#cluster_1#"]
        )

        failed.assert_not_called()
        np.testing.assert_array_equal(viewer.current_colors[0], mcolors.to_rgba("blue"))
        succeeded.assert_called_once_with(
            viewer,
            "Applied: 1 node (blue) "
            "Skipped 2 expressions with no color, scale or shape: #noise#, #cluster_1#.",
        )

    def test_the_skip_is_reported_when_nothing_else_matched_too(self):
        viewer = self.make_viewer()
        succeeded, failed = self.run_color(
            viewer, ["{Length>500}", "red", "#cluster_1#"]
        )

        failed.assert_not_called()
        succeeded.assert_called_once_with(
            viewer,
            "No nodes matched criteria. "
            "Skipped 1 expression with no color, scale or shape: #cluster_1#.",
        )

    def test_a_command_without_skipped_expressions_reports_as_before(self):
        viewer = self.make_viewer()
        succeeded, failed = self.run_color(viewer, ['"node"', "red"])

        failed.assert_not_called()
        succeeded.assert_called_once_with(viewer, "Applied: 1 node (red)")


class CysteinePortalTests(PortalFixture, unittest.TestCase):
    def test_cysteine_manual_portal_parity(self):
        manual = add_alignment(Viewer(self.directory.name))
        add_alignment(self.viewer)
        manual.selected_indices = self.viewer.selected_indices = [1]
        manual.process_command('color C53 red 2x')
        result = self.finish(self.portal.submit('cysteine', 'color C53 red 2x')['request_id'])
        self.assertEqual(result['status'], 'succeeded', result)
        np.testing.assert_array_equal(manual.current_colors, self.viewer.current_colors)
        np.testing.assert_array_equal(manual.current_sizes, self.viewer.current_sizes)
        self.assertEqual(self.viewer.current_sizes[0], cfg.NODE_SIZE * 2)
        np.testing.assert_array_equal(self.viewer.current_colors[0], mcolors.to_rgba('red'))
        np.testing.assert_array_equal(self.viewer.current_colors[1:], np.ones((2, 4)))


if __name__ == "__main__":
    unittest.main()
