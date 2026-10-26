"""The color command (commands/color.py): residue selections versus Matplotlib
cycle names, scales, alphas and shapes, attribute sets, and batches applied
all-or-nothing."""
import io
import itertools
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
            "Skipped 1 expression with no color, alpha, scale or shape: #cluster_1#."
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
            "Skipped 2 expressions with no color, alpha, scale or shape: #noise#, #cluster_1#.",
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
            "Skipped 1 expression with no color, alpha, scale or shape: #cluster_1#.",
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


def target_viewer(alphas=(1.0, 0.4, 1.0, 0.7, 0.4)):
    """Five visible nodes, each with its own alpha, for the targets of color.

    Nodes 0 and 3 are in cluster 1, nodes 1 and 2 in cluster 2 and node 4 is
    noise. Their Length is 100, 600, 300, 700 and 50. Every node starts
    with the color (0.1, 0.2, 0.3), size 5 and no shape of its own.
    """
    viewer = one_node_viewer()
    viewer.n_nodes = 5
    viewer.full_headers = ["ATA_a", "ATA_b", "XYZ_c", "XYZ_d", "ATA_e"]
    viewer.cluster_labels = np.array([1, 2, 2, 1, -1])
    viewer.group_labels = [set() for _ in range(5)]
    viewer.metadata = {
        "Length": {"type": "number", "values": np.array([100.0, 600.0, 300.0, 700.0, 50.0])}
    }
    viewer.current_colors = np.tile([0.1, 0.2, 0.3, 1.0], (5, 1))
    viewer.current_colors[:, 3] = alphas
    viewer.current_sizes = np.full(5, 5.0)
    viewer.visible_mask = np.ones(5, dtype=bool)
    return viewer


def run_line(viewer, line):
    """Run color with the words of line, as the command line splits it at spaces.

    Returns (succeeded, failed) as reported_outcomes records them.
    """
    with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
        color_command.run(viewer, line.split())
    return succeeded, failed


def snapshot(viewer):
    shapes = viewer.current_shapes.copy() if hasattr(viewer, "current_shapes") else None
    return viewer.current_colors.copy(), viewer.current_sizes.copy(), shapes


def rgba(name, alpha=None):
    """The RGBA of a color name, with its alpha replaced when one is given."""
    value = list(mcolors.to_rgba(name))
    if alpha is not None:
        value[3] = alpha
    return value


class ColorAttributeSetTests(unittest.TestCase):
    """A set <ITEM,ITEM,...> or <ITEM ITEM ...> gives one target its attributes."""

    def test_every_way_of_writing_a_set_gives_the_same_result(self):
        forms = [
            "#cluster_1# <red,50%,star>",
            "#cluster_1# <star,red,50%>",
            "#cluster_1#<red,50%,star>",
            "#cluster_1# <red, 50%, star>",
            "#cluster_1# <red , 50% ,star>",
            "#cluster_1# <red 50% star>",
            "#cluster_1#<red 50% star>",
            "#cluster_1# < red 50% star >",
            "#cluster_1#< red, 50%, star >",
            "#cluster_1# red 50% star",
        ]
        for form in forms:
            with self.subTest(form=form):
                viewer = target_viewer()
                succeeded, failed = run_line(viewer, form)

                failed.assert_not_called()
                succeeded.assert_called_once_with(viewer, "Applied: 2 nodes (red, 50%, star)")
                for node in (0, 3):
                    np.testing.assert_array_equal(viewer.current_colors[node], rgba("red", 0.5))
                    self.assertEqual(viewer.current_shapes[node], "star")
                for node in (1, 2, 4):
                    self.assertEqual(viewer.current_shapes[node], "disc")
                    np.testing.assert_array_equal(viewer.current_colors[node, :3], [0.1, 0.2, 0.3])
                np.testing.assert_array_equal(viewer.current_colors[:, 3], [0.5, 0.4, 1.0, 0.5, 0.4])
                viewer._save_state.assert_called_once_with()

    def test_the_items_may_come_in_any_order_with_either_separator(self):
        for order in itertools.permutations(["red", "50%", "2x", "star"]):
            for separator in (",", " "):
                with self.subTest(order=order, separator=separator):
                    viewer = target_viewer()
                    succeeded, failed = run_line(viewer, f"#cluster_1# <{separator.join(order)}>")

                    failed.assert_not_called()
                    np.testing.assert_array_equal(viewer.current_colors[0], rgba("red", 0.5))
                    self.assertEqual(viewer.current_sizes[0], cfg.NODE_SIZE * 2)
                    self.assertEqual(viewer.current_shapes[0], "star")
                    self.assertEqual(viewer.current_sizes[1], 5.0)

    def test_a_set_may_hold_one_to_four_items(self):
        cases = [
            ("<red>", rgba("red", 1.0), 5.0, "disc"),
            ("<30%>", [0.1, 0.2, 0.3, 0.3], 5.0, "disc"),
            ("<3x>", [0.1, 0.2, 0.3, 1.0], cfg.NODE_SIZE * 3, "disc"),
            ("<square>", [0.1, 0.2, 0.3, 1.0], 5.0, "square"),
            ("<blue,30%>", rgba("blue", 0.3), 5.0, "disc"),
            ("<blue,30%,3x>", rgba("blue", 0.3), cfg.NODE_SIZE * 3, "disc"),
            ("<blue,30%,3x,square>", rgba("blue", 0.3), cfg.NODE_SIZE * 3, "square"),
        ]
        for text, expected_color, expected_size, expected_shape in cases:
            with self.subTest(text=text):
                viewer = target_viewer(alphas=(1.0,) * 5)
                succeeded, failed = run_line(viewer, f"#cluster_1# {text}")

                failed.assert_not_called()
                np.testing.assert_array_equal(viewer.current_colors[0], expected_color)
                self.assertEqual(viewer.current_sizes[0], expected_size)
                self.assertEqual(viewer.current_shapes[0], expected_shape)

    def test_a_set_alone_targets_the_selected_nodes(self):
        viewer = target_viewer(alphas=(1.0,) * 5)
        viewer.selected_indices = [1, 2]
        succeeded, failed = run_line(viewer, "<80%,square>")

        failed.assert_not_called()
        succeeded.assert_called_once_with(viewer, "Applied: 2 nodes (80%, square)")
        np.testing.assert_array_equal(viewer.current_colors[:, 3], [1.0, 0.8, 0.8, 1.0, 1.0])
        self.assertEqual(list(viewer.current_shapes), ["disc", "square", "square", "disc", "disc"])

    def test_an_expression_after_a_set_starts_a_new_target(self):
        # A set belongs to the expression before it, and with none, to the selection.
        viewer = target_viewer(alphas=(1.0,) * 5)
        viewer.selected_indices = [4]
        succeeded, failed = run_line(viewer, "<red> #cluster_1# <blue>")

        failed.assert_not_called()
        np.testing.assert_array_equal(viewer.current_colors[4], rgba("red"))
        np.testing.assert_array_equal(viewer.current_colors[0], rgba("blue"))
        np.testing.assert_array_equal(viewer.current_colors[1, :3], [0.1, 0.2, 0.3])

    def test_a_set_before_an_expression_does_not_belong_to_it(self):
        # Loose items may come before their expression; a set belongs to the one before it.
        viewer = target_viewer(alphas=(1.0,) * 5)
        viewer.selected_indices = [4]
        succeeded, failed = run_line(viewer, "<red> #cluster_1#")

        failed.assert_not_called()
        np.testing.assert_array_equal(viewer.current_colors[4], rgba("red"))
        np.testing.assert_array_equal(viewer.current_colors[0, :3], [0.1, 0.2, 0.3])
        self.assertEqual(
            str(succeeded.call_args.args[1]),
            "Applied: 1 node (red) "
            "Skipped 1 expression with no color, alpha, scale or shape: #cluster_1#.",
        )

    def test_several_targets_each_take_their_own_set(self):
        viewer = target_viewer()
        succeeded, failed = run_line(viewer, '"ATA*"&#cluster_2# <blue,1.5x> #noise# <20%>')

        failed.assert_not_called()
        succeeded.assert_called_once_with(
            viewer, "Applied: 1 node (blue, 1.5x); 1 node (20%)"
        )
        np.testing.assert_array_equal(viewer.current_colors[1], rgba("blue", 0.4))
        self.assertEqual(viewer.current_sizes[1], cfg.NODE_SIZE * 1.5)
        np.testing.assert_array_equal(viewer.current_colors[4], [0.1, 0.2, 0.3, 0.2])
        self.assertEqual(viewer.current_sizes[4], 5.0)
        # Nothing else changed.
        np.testing.assert_array_equal(
            viewer.current_colors[[0, 2, 3]], [[0.1, 0.2, 0.3, alpha] for alpha in (1.0, 1.0, 0.7)]
        )
        viewer._save_state.assert_called_once_with()

    def test_sets_and_loose_items_may_follow_each_other_across_targets(self):
        viewer = target_viewer(alphas=(1.0,) * 5)
        run_line(viewer, "#cluster_1# <red,2x> #cluster_2# blue 50% #noise# <0x>")

        np.testing.assert_array_equal(viewer.current_colors[0], rgba("red"))
        np.testing.assert_array_equal(viewer.current_colors[1], rgba("blue", 0.5))
        self.assertEqual(viewer.current_sizes[4], 0.0)
        self.assertEqual(viewer.current_sizes[0], cfg.NODE_SIZE * 2)

    def test_every_shape_name_and_symbol_is_an_item(self):
        names = {
            "circle": "disc", "triangle": "triangle_up", "square": "square", "star": "star",
            "x": "x", "X": "x", "vbar": "vbar", "hbar": "hbar", "diamond": "diamond",
            "^": "^", "|": "|", "v": "v", "*": "*", "+": "+", "++": "++", "-": "-",
            "->": "->", ">": ">", "o": "o", "s": "s", "arrow": "arrow", "tailed_arrow": "tailed_arrow",
        }
        for written, shape in names.items():
            for form in ("#cluster_1# <red,{}>", "#cluster_1# <{},red>", "#cluster_1# <red {}>"):
                with self.subTest(shape=written, form=form):
                    viewer = target_viewer()
                    succeeded, failed = run_line(viewer, form.format(written))

                    failed.assert_not_called()
                    self.assertEqual(viewer.current_shapes[0], shape)
                    np.testing.assert_array_equal(viewer.current_colors[0, :3], rgba("red")[:3])

    def test_a_shape_that_ends_in_the_closing_bracket_closes_a_space_separated_set(self):
        # > and -> end a word with >, so a space-separated set closes at the first such word.
        viewer = target_viewer()
        succeeded, failed = run_line(viewer, "#cluster_1# <red > star>")

        failed.assert_called_once()
        self.assertIn("'star>'", str(failed.call_args.args[1]))
        succeeded.assert_not_called()
        viewer._save_state.assert_not_called()

    def test_hex_colors_with_an_alpha_set_both(self):
        cases = {"#ff000080": 128 / 255, "#f008": 0x88 / 255, "#ff0000ff": 1.0, "#FF000000": 0.0, "none": 0.0}
        for color_text, alpha in cases.items():
            for form in ("#cluster_1# <{}>", "#cluster_1# {}"):
                with self.subTest(color=color_text, form=form):
                    viewer = target_viewer()
                    succeeded, failed = run_line(viewer, form.format(color_text))

                    failed.assert_not_called()
                    expected = [0.0, 0.0, 0.0] if color_text == "none" else [1.0, 0.0, 0.0]
                    np.testing.assert_array_equal(viewer.current_colors[0], [*expected, alpha])
                    self.assertEqual(succeeded.call_args.args[1], f"Applied: 2 nodes ({color_text})")

    def test_a_color_in_a_set_is_never_read_as_an_expression(self):
        # In the loose form C53 selects residue C at position 53; a set holds no expression.
        viewer = target_viewer()
        succeeded, failed = run_line(viewer, "#cluster_1# <C53>")

        failed.assert_not_called()
        np.testing.assert_array_equal(viewer.current_colors[0, :3], mcolors.to_rgb("C53"))

    def test_a_set_never_classifies_as_an_expression(self):
        kinds = color_command.Command_Engine.SelectionClassificationKind
        classify = color_command.Command_Engine.classify_selection_expression
        # The set a command is usually given is no expression at all.
        self.assertEqual(classify("<red,50%,star>").kind, kinds.NOT_EXPRESSION)
        for text in [
            "<red,50%,star>", "<red", "50%>", "<>", "<^>", "<|,red>", "<red,>>", "<#ff0000,50%>",
            "<red,50%,star", "#c#<red>", "{Length<500}<red>", "<C53>", "<x2>",
        ]:
            with self.subTest(text=text):
                self.assertNotEqual(classify(text).kind, kinds.VALID_EXPRESSION)

    def test_the_loose_form_still_works(self):
        forms = {
            "#cluster_1# red 2x star": ([1.0, 0.0, 0.0, 1.0], 2, "star"),
            "red #cluster_1# 2x star": ([1.0, 0.0, 0.0, 1.0], 2, "star"),
            "#cluster_1# red 50% 2x star": ([1.0, 0.0, 0.0, 0.5], 2, "star"),
            "#cluster_1# 2x star 50% red": ([1.0, 0.0, 0.0, 0.5], 2, "star"),
        }
        for form, (expected_color, factor, shape) in forms.items():
            with self.subTest(form=form):
                viewer = target_viewer(alphas=(1.0,) * 5)
                succeeded, failed = run_line(viewer, form)

                failed.assert_not_called()
                np.testing.assert_array_equal(viewer.current_colors[0], expected_color)
                self.assertEqual(viewer.current_sizes[0], cfg.NODE_SIZE * factor)
                self.assertEqual(viewer.current_shapes[0], shape)

    def test_the_loose_form_still_chains_targets_and_gives_the_old_report(self):
        viewer = target_viewer(alphas=(1.0,) * 5)
        succeeded, failed = run_line(viewer, "#cluster_1# red #noise# 0x")

        failed.assert_not_called()
        succeeded.assert_called_once_with(viewer, "Applied: 2 nodes (red); 1 node (0.0x)")


class ColorSetSplittingTests(unittest.TestCase):
    """A set glued to its expression is split off at its own <, never at one inside the expression."""

    def test_the_glued_set_is_split_off_the_expression(self):
        split = color_command._split_at_set
        cases = {
            "#cluster_1#<red>": ("#cluster_1#", "<red>"),
            "#cluster_1#<red": ("#cluster_1#", "<red"),
            "<red,50%>": (None, "<red,50%>"),
            "<": (None, "<"),
            "(RHK)71<red>": ("(RHK)71", "<red>"),
            "K106&#noise#<red,2x>": ("K106&#noise#", "<red,2x>"),
            "$sele$<red>": ("$sele$", "<red>"),
            "{Length<500}<red>": ("{Length<500}", "<red>"),
            "{Length<500}&{Length>100}<red>": ("{Length<500}&{Length>100}", "<red>"),
            '"A<B"<red>': ('"A<B"', "<red>"),
            '"A<B"&#cluster_1#<red>': ('"A<B"&#cluster_1#', "<red>"),
            "@a<b.txt@<red>": ("@a<b.txt@", "<red>"),
            "#a<b#<red>": ("#a<b#", "<red>"),
            "{Name=*#<*}<red>": ("{Name=*#<*}", "<red>"),
            "<red>#cluster_1#": (None, "<red>#cluster_1#"),
        }
        for arg, expected in cases.items():
            with self.subTest(arg=arg):
                self.assertEqual(split(arg), expected)

    def test_a_less_than_sign_inside_delimiters_never_opens_a_set(self):
        split = color_command._split_at_set
        for arg in [
            "{Length<500}", '"A<B"', "@a<b.txt@", "#a<b#", "{Length<500}&{Length<900}",
            "{Length<500", '"A<B', "red", "#ff0000", "2x", "->", ">", "",
        ]:
            with self.subTest(arg=arg):
                self.assertEqual(split(arg), (arg, None))

    def test_the_words_of_a_set_are_joined_up_to_the_word_that_closes_it(self):
        join = color_command._join_sets
        cases = [
            (["#c#", "<red", "50%", "star>", "#d#", "<blue>"], ["#c#", "<red 50% star>", "#d#", "<blue>"]),
            (["#c#<red", "50%>", "blue"], ["#c#", "<red 50%>", "blue"]),
            (["<", "red", ">"], ["< red >"]),
            (["<red,", "50%>"], ["<red, 50%>"]),
            (["<red", ">>", "x"], ["<red >>", "x"]),
            (["{Length<500}<red", "50%>"], ["{Length<500}", "<red 50%>"]),
            # Never closed: it takes every word to the end.
            (["#c#", "<red", "50%", "#d#", "blue"], ["#c#", "<red 50% #d# blue"]),
            (["red", "2x"], ["red", "2x"]),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(join(args), expected)

    def test_the_command_line_hands_a_glued_space_separated_set_over_in_pieces(self):
        # The engine splits the line at spaces: #cluster_1#<red 50%> arrives as two words.
        self.assertEqual("color #cluster_1#<red 50%>".split()[1:], ["#cluster_1#<red", "50%>"])
        viewer = target_viewer()
        succeeded, failed = run_line(viewer, "#cluster_1#<red 50%>")

        failed.assert_not_called()
        np.testing.assert_array_equal(viewer.current_colors[0], rgba("red", 0.5))

    def test_metadata_with_a_less_than_sign_is_the_expression_of_a_glued_set(self):
        for line in [
            "{Length<500}<red>", "{Length<500} <red>", "{Length<500}<red 50%>", "{Length<500}<red,50%>",
            "{Length<500}&#cluster_1#<red,50%>",
        ]:
            with self.subTest(line=line):
                viewer = target_viewer(alphas=(1.0,) * 5)
                succeeded, failed = run_line(viewer, line)

                failed.assert_not_called()
                reddened = [node for node in range(5) if np.array_equal(viewer.current_colors[node, :3], [1, 0, 0])]
                self.assertEqual(reddened, [0] if "cluster_1" in line else [0, 2, 4])

    def test_header_text_with_a_less_than_sign_is_the_expression_of_a_glued_set(self):
        for line in ['"XYZ<c"<red>', '"XYZ<c" <red>', '"XYZ<c"<red 50%>', '"XYZ*"&"*<c"<red>']:
            with self.subTest(line=line):
                viewer = target_viewer(alphas=(1.0,) * 5)
                viewer.full_headers[2] = "XYZ<c"
                succeeded, failed = run_line(viewer, line)

                failed.assert_not_called()
                np.testing.assert_array_equal(viewer.current_colors[2, :3], [1, 0, 0])
                np.testing.assert_array_equal(viewer.current_colors[[0, 1, 3, 4], :3], [[0.1, 0.2, 0.3]] * 4)

    def test_the_loose_form_reads_header_text_with_a_less_than_sign(self):
        viewer = target_viewer(alphas=(1.0,) * 5)
        viewer.full_headers[2] = "XYZ<c"
        succeeded, failed = run_line(viewer, '"XYZ<c" red')

        failed.assert_not_called()
        np.testing.assert_array_equal(viewer.current_colors[2, :3], [1, 0, 0])

    def test_an_unclosed_expression_before_a_less_than_sign_is_the_expressions_error(self):
        viewer = target_viewer()
        before = snapshot(viewer)
        succeeded, failed = run_line(viewer, "{Length<500<red>")

        failed.assert_called_once()
        succeeded.assert_not_called()
        np.testing.assert_array_equal(viewer.current_colors, before[0])
        viewer._save_state.assert_not_called()


class ColorSetRefusalTests(unittest.TestCase):
    """What cannot be read is refused whole: nothing changes, and no undo step is saved."""

    def assert_refused(self, line):
        """Run line, check it changed nothing, and return the message that refused it."""
        viewer = target_viewer()
        before = snapshot(viewer)
        succeeded, failed = run_line(viewer, line)

        failed.assert_called_once()
        succeeded.assert_not_called()
        message = str(failed.call_args.args[1])
        np.testing.assert_array_equal(viewer.current_colors, before[0], line)
        np.testing.assert_array_equal(viewer.current_sizes, before[1], line)
        self.assertIsNone(before[2])
        self.assertFalse(hasattr(viewer, "current_shapes"), line)
        viewer._save_state.assert_not_called()
        viewer.promote_nodes.assert_not_called()
        viewer.update_nodes.assert_not_called()
        return message

    def test_an_empty_set_is_refused(self):
        for text in ("<>", "< >", "<  >"):
            with self.subTest(text=text):
                message = self.assert_refused(f"#cluster_1# {text}")
                self.assertTrue(message.startswith("Error: The attribute set '"), message)
                self.assertIn("is empty", message)
        self.assertEqual(
            self.assert_refused("#cluster_1# <>"),
            "Error: The attribute set '<>' is empty. A set holds 1 to 4 attributes, "
            "e.g. <red,50%,star> or <red 50% star>.",
        )

    def test_two_items_of_one_kind_are_refused(self):
        cases = {
            "<red,blue>": ("red", "blue"),
            "<red blue>": ("red", "blue"),
            "<50%,20%>": ("50%", "20%"),
            "<2x,3x>": ("2x", "3x"),
            "<star,disc>": ("star", "disc"),
            "<red,2x,blue>": ("red", "blue"),
            "<star,red,50%,circle>": ("star", "circle"),
        }
        for text, (first, second) in cases.items():
            with self.subTest(text=text):
                self.assertEqual(
                    self.assert_refused(f"#cluster_1# {text}"),
                    f"Error: The attribute set '{text}' has two items of one kind, '{first}' and "
                    f"'{second}'. A set holds at most one color, one alpha, one scale and one shape.",
                )

    def test_an_unknown_item_is_refused_by_name(self):
        for text, item in {
            "<red,blu>": "blu", "<blu>": "blu", "<red 50 star>": "50", "<red,50>": "50",
            "<red,x2>": "x2", "<red,2X>": "2X", "<red,%>": "%", "<red,abc%>": "abc%", "<red,1_0%>": "1_0%",
            "<red,circles>": "circles",
        }.items():
            with self.subTest(text=text):
                self.assertEqual(
                    self.assert_refused(f"#cluster_1# {text}"),
                    f"Error: Unknown attribute '{item}' in the set '{text}'. Expected a color, "
                    "an alpha such as 50%, a scale such as 2x, or a shape.",
                )

    def test_a_malformed_set_is_refused(self):
        for line, name in {
            "#cluster_1# <red,50%": "<red,50%",
            "#cluster_1# red,50%>": "red,50%>",
            "#cluster_1# <red,,50%>": "<red,,50%>",
            "#cluster_1# <,red>": "<,red>",
            "#cluster_1# <red,>": "<red,>",
            "#cluster_1# <,>": "<,>",
            "#cluster_1# <red,50%,star": "<red,50%,star",
            "#cluster_1# red,50%": "red,50%",
            "#cluster_1# <": "<",
            "#cluster_1# <red <blue>": "<red <blue>",
            "#cluster_1# <red,<blue>": "<red,<blue>",
            "#cluster_1# <red <blue 50%>": "<red <blue 50%>",
            "#cluster_1# <red>x": "<red>x",
            "#cluster_1# <red>#noise#": "<red>#noise#",
            "#cluster_1# <red>#noise# blue": "<red>#noise# blue",
            "#cluster_1#<red><blue>": "<red><blue>",
            "#cluster_1#<red,50% #noise# 2x": "<red,50% #noise# 2x",
        }.items():
            with self.subTest(line=line):
                self.assertEqual(
                    self.assert_refused(line),
                    f"Error: Malformed attribute set '{name}'. A set is <ITEM,ITEM,...> or "
                    "<ITEM ITEM ...>: opened by <, closed by > and holding no empty item "
                    "and no other <.",
                )

    def test_a_set_that_never_closes_is_refused_whatever_follows(self):
        for line in [
            "#cluster_1# <red 50%", "#cluster_1# <red 50% #noise# blue", "#cluster_1#<red 50%",
            "<red 50%", "<red", "<",
        ]:
            with self.subTest(line=line):
                self.assertIn("Malformed attribute set", self.assert_refused(line))

    def test_a_set_that_mixes_commas_and_spaces_is_refused(self):
        for line, name in {
            "#cluster_1# <red, 50% star>": "<red, 50% star>",
            "#cluster_1# <red 50%, star>": "<red 50%, star>",
            "#cluster_1# <red,50% star>": "<red,50% star>",
            "#cluster_1# <red 50%,star>": "<red 50%,star>",
            "#cluster_1# <red, 50%, star 2x>": "<red, 50%, star 2x>",
            "#cluster_1# <red 50% , star>": "<red 50% , star>",
            "#cluster_1#<red, 50% star>": "<red, 50% star>",
        }.items():
            with self.subTest(line=line):
                self.assertEqual(
                    self.assert_refused(line),
                    f"Error: The attribute set '{name}' mixes commas and spaces between its items. "
                    "Separate them all with commas (<red,50%,star>) or all with spaces (<red 50% star>).",
                )

    def test_loose_items_and_a_set_for_one_target_are_refused(self):
        for line, loose, text in [
            ("#cluster_1# red <50%>", "red", "<50%>"),
            ("#cluster_1# 2x <red>", "2x", "<red>"),
            ("#cluster_1# <50%> red", "red", "<50%>"),
            ("#cluster_1# <red,50%> star", "star", "<red,50%>"),
            ("#cluster_1# red 2x <50% star>", "red", "<50% star>"),
            ("red <50%>", "red", "<50%>"),
            ("<50%> red", "red", "<50%>"),
            ("red #cluster_1# <50%>", "red", "<50%>"),
            ("#cluster_1#<red> blue", "blue", "<red>"),
        ]:
            with self.subTest(line=line):
                self.assertEqual(
                    self.assert_refused(line),
                    f"Error: '{loose}' and the attribute set '{text}' are given for one target. "
                    "Give a target its attributes either all loose (red 50% star) "
                    "or all in one set (<red,50%,star>).",
                )

    def test_two_sets_for_one_target_are_refused(self):
        for line, first, second in [
            ("#cluster_1# <red> <blue>", "<red>", "<blue>"),
            ("#cluster_1# <red> <50%>", "<red>", "<50%>"),
            ("<red> <50%>", "<red>", "<50%>"),
            ("#cluster_1#<red> <blue 2x>", "<red>", "<blue 2x>"),
        ]:
            with self.subTest(line=line):
                self.assertEqual(
                    self.assert_refused(line),
                    f"Error: The attribute sets '{first}' and '{second}' are given for one target. "
                    "Give a target one set, and start the next target with an expression.",
                )

    def test_two_alphas_for_one_target_are_refused(self):
        for line, first, second in [
            ("#cluster_1# <#ff000080,50%>", "#ff000080", "50%"),
            ("#cluster_1# <50%,#ff000080>", "50%", "#ff000080"),
            ("#cluster_1# <#f008 50%>", "#f008", "50%"),
            ("#cluster_1# <none,50%>", "none", "50%"),
            ("#cluster_1# #ff000080 50%", "#ff000080", "50%"),
            ("#cluster_1# 50% #ff000080", "50%", "#ff000080"),
            ("#cluster_1# red 50% 20%", "50%", "20%"),
            ("#cluster_1# 50% 20% red", "50%", "20%"),
        ]:
            with self.subTest(line=line):
                self.assertEqual(
                    self.assert_refused(line),
                    f"Error: '{first}' and '{second}' both set the alpha of one target. "
                    "Give the alpha once, either in an 8-digit hex color or as N%.",
                )

    def test_an_alpha_outside_zero_to_one_hundred_is_refused(self):
        for token in ["-5%", "-0.1%", "100.1%", "101%", "1000%", "nan%", "inf%", "-inf%", "1e400%", "1e3%"]:
            for form in ("#cluster_1# red {}", "#cluster_1# {}", "#cluster_1# <red,{}>", "#cluster_1# <{}>"):
                with self.subTest(token=token, form=form):
                    self.assertEqual(
                        self.assert_refused(form.format(token)),
                        f"Error: Invalid alpha '{token}'. An alpha is a number from 0 to 100 "
                        "followed by %, e.g. 50%, 12.5% or 0%.",
                    )

    def test_a_scale_that_cannot_be_used_is_refused_in_a_set_too(self):
        for token in ["-2x", "nanx", "infx", "1e400x"]:
            with self.subTest(token=token):
                self.assertIn(
                    f"Invalid scale '{token}'", self.assert_refused(f"#cluster_1# <red,{token}>")
                )
        self.assertEqual(
            self.assert_refused("#cluster_1# <red 1e300x>"),
            "Error: Scale '1e300x' is too large to make a valid node size.",
        )

    def test_a_loose_argument_that_is_nothing_is_refused_by_name(self):
        for token in ["%", "abc%", "red,50%", "blu"]:
            with self.subTest(token=token):
                self.assertIn(f"'{token}'", self.assert_refused(f"#cluster_1# {token}"))

    def test_a_bad_set_after_good_targets_changes_nothing(self):
        for line in [
            "#cluster_1# <red,50%> #noise# <blu>",
            "#cluster_1# <red,50%> #noise# <>",
            "#cluster_1# <red,50%> #noise# <20%",
            "#cluster_1# <red,50%> #noise# <red,blue>",
            "#cluster_1# <red,50%> #noise# red <20%>",
            "#cluster_1# <red,50%> #noise# <red, 20% star>",
            "#cluster_1# <red,50%> #cluster_9# <blue>",
            "#cluster_1# red 50% #noise# <20%> <10%>",
            "#cluster_1# <red,50%> #noise# 20% 10%",
            "#cluster_1# <red,50%,2x,star> #noise# <-2x>",
        ]:
            with self.subTest(line=line):
                self.assert_refused(line)


class ColorAlphaTests(unittest.TestCase):
    """Color and alpha are independent: each changes only what is given."""

    def run_on_cluster_two(self, line, alphas=(1.0, 0.4, 1.0, 0.7, 0.4)):
        """Cluster 2 is node 1 (alpha 0.4) and node 2 (alpha 1.0)."""
        viewer = target_viewer(alphas)
        succeeded, failed = run_line(viewer, line)
        failed.assert_not_called()
        return viewer, succeeded

    def test_a_color_alone_keeps_each_nodes_alpha(self):
        for form in ("#cluster_2# red", "#cluster_2# <red>", "#cluster_2# <#ff0000>"):
            with self.subTest(form=form):
                viewer, succeeded = self.run_on_cluster_two(form)
                np.testing.assert_array_equal(viewer.current_colors[1], rgba("red", 0.4))
                np.testing.assert_array_equal(viewer.current_colors[2], rgba("red", 1.0))
                np.testing.assert_array_equal(viewer.current_colors[0], [0.1, 0.2, 0.3, 1.0])

    def test_recoloring_a_translucent_node_keeps_it_translucent_until_100_percent_is_given(self):
        viewer, _ = self.run_on_cluster_two("#noise# <red>")
        self.assertEqual(viewer.current_colors[4, 3], 0.4)
        run_line(viewer, "#noise# <100%>")
        np.testing.assert_array_equal(viewer.current_colors[4], rgba("red", 1.0))

    def test_an_alpha_alone_keeps_each_nodes_color(self):
        for form in ("#cluster_2# 50%", "#cluster_2# <50%>"):
            with self.subTest(form=form):
                viewer, succeeded = self.run_on_cluster_two(form)
                np.testing.assert_array_equal(viewer.current_colors[1], [0.1, 0.2, 0.3, 0.5])
                np.testing.assert_array_equal(viewer.current_colors[2], [0.1, 0.2, 0.3, 0.5])
                self.assertEqual(succeeded.call_args.args[1], "Applied: 2 nodes (50%)")

    def test_a_color_and_an_alpha_set_both(self):
        for form in ("#cluster_2# red 50%", "#cluster_2# <red,50%>", "#cluster_2# <50% red>"):
            with self.subTest(form=form):
                viewer, succeeded = self.run_on_cluster_two(form)
                for node in (1, 2):
                    np.testing.assert_array_equal(viewer.current_colors[node], rgba("red", 0.5))

    def test_an_eight_digit_hex_sets_the_color_and_the_alpha(self):
        viewer, succeeded = self.run_on_cluster_two("#cluster_2# <#00ff0040>")
        for node in (1, 2):
            np.testing.assert_array_equal(viewer.current_colors[node], [0.0, 1.0, 0.0, 64 / 255])

    def test_the_alpha_range_ends_and_fractions_are_accepted(self):
        for token, percent in {
            "0%": 0, "100%": 100, "12.5%": 12.5, "0.0%": 0, "-0%": 0, "+50%": 50,
            "1e1%": 10, "99.9%": 99.9, "50.%": 50, ".5%": 0.5,
        }.items():
            for form in ("#cluster_2# {}", "#cluster_2# <{}>"):
                with self.subTest(token=token, form=form):
                    viewer, _ = self.run_on_cluster_two(form.format(token))
                    self.assertAlmostEqual(viewer.current_colors[1, 3], percent / 100, places=12)
                    self.assertFalse(np.signbit(viewer.current_colors[1, 3]))

    def test_the_report_lists_the_alpha(self):
        viewer, succeeded = self.run_on_cluster_two("#cluster_2# <red,50%,2x,star>")
        self.assertEqual(succeeded.call_args.args[1], "Applied: 2 nodes (red, 50%, 2.0x, star)")
        viewer, succeeded = self.run_on_cluster_two("#cluster_2# <12.5%>")
        self.assertEqual(succeeded.call_args.args[1], "Applied: 2 nodes (12.5%)")

    def test_float32_colors_keep_what_they_are_not_given(self):
        # The Viewer keeps its colors as float32.
        viewer = target_viewer()
        viewer.current_colors = viewer.current_colors.astype(np.float32)
        run_line(viewer, "#cluster_2# <red>")
        np.testing.assert_array_equal(
            viewer.current_colors[1], np.array(rgba("red", 0.4), dtype=np.float32)
        )
        run_line(viewer, "#cluster_2# <30%>")
        np.testing.assert_array_equal(
            viewer.current_colors[2], np.array(rgba("red", 0.3), dtype=np.float32)
        )

    def test_hidden_nodes_keep_their_alpha_and_color(self):
        viewer = target_viewer()
        viewer.visible_mask[1] = False
        run_line(viewer, "#cluster_2# <red,10%>")
        np.testing.assert_array_equal(viewer.current_colors[1], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_equal(viewer.current_colors[2], rgba("red", 0.1))

    def test_a_command_with_several_targets_is_one_undo_step_and_one_redraw(self):
        viewer = target_viewer()
        run_line(viewer, "#cluster_2# <10%> #noise# <20%>")
        viewer._save_state.assert_called_once_with()
        viewer.update_nodes.assert_called_once_with()
        viewer.promote_nodes.assert_called_once()


class ColorAlphaPortalTests(PortalFixture, unittest.TestCase):
    """The command as a typed or MCP command line reaches it: split at spaces."""

    def test_a_space_separated_glued_set_reaches_color_in_pieces_through_the_portal(self):
        manual = Viewer(self.directory.name)
        manual.process_command('color #cluster_0#<red 50% 2x>')
        result = self.finish(self.portal.submit('set', 'color #cluster_0#<red 50% 2x>')['request_id'])

        self.assertEqual(result['status'], 'succeeded', result)
        np.testing.assert_array_equal(manual.current_colors, self.viewer.current_colors)
        np.testing.assert_array_equal(self.viewer.current_colors[:2], [rgba('red', 0.5)] * 2)
        np.testing.assert_array_equal(self.viewer.current_colors[2], np.ones(4))
        self.assertEqual(list(self.viewer.current_sizes), [cfg.NODE_SIZE * 2] * 2 + [1.0])

    def test_color_reset_restores_the_alpha_too(self):
        viewer = Viewer(self.directory.name)
        viewer.process_command('color #cluster_0# <red,25%>')
        np.testing.assert_array_equal(viewer.current_colors[0], rgba('red', 0.25))
        viewer.process_command('color reset')
        np.testing.assert_array_equal(
            viewer.current_colors, np.tile(mcolors.to_rgba(cfg.INITIAL_NODE_COLOR), (3, 1))
        )


if __name__ == "__main__":
    unittest.main()
