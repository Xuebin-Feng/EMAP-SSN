"""The spectrum command (commands/spectrum.py): argument roles, the colors it
applies, and its min/max legend on the terminal and the console overlay."""
import io
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands import spectrum
from tests.command_fixtures import add_alignment
from tests.viewer_fixtures import Viewer


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
        # spectrum ends by running `meta display` for the property when the
        # viewer has the metadata HUD; only that call is replaced, so every
        # other import stays real.
        with mock.patch("commands.meta.run"), mock.patch.object(sys, "stdout", output):
            spectrum.run(viewer, args)

    def test_the_property_goes_to_the_metadata_hud_only_when_there_is_one(self):
        # The desktop viewer keeps hud_displays; the VR viewer and other
        # headless viewers have no HUD, so there is nothing to display it on.
        for has_hud in (True, False):
            with self.subTest(has_hud=has_hud):
                viewer = self.make_viewer([1.0, 3.0])
                if has_hud:
                    viewer.hud_displays = {}
                with mock.patch("commands.meta.run") as meta_run,                         mock.patch.object(sys, "stdout", io.StringIO()):
                    spectrum.run(viewer, ["{Length}"])
                if has_hud:
                    meta_run.assert_called_once_with(viewer, ["display", "Length"])
                else:
                    meta_run.assert_not_called()
                self.assertIn("Spectrum coloring applied", viewer.console_text.text)

    def test_a_viewer_without_the_hud_prints_no_warning_about_it(self):
        viewer = self.make_viewer([1.0, 3.0])
        output = io.StringIO()
        with mock.patch.object(sys, "stdout", output):
            spectrum.run(viewer, ["{Length}"])
        self.assertIn("Spectrum coloring applied", output.getvalue())
        self.assertNotIn("metadata display", output.getvalue())

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

    def test_infinite_values_are_gray_and_do_not_stretch_the_range(self):
        viewer = self.make_viewer([350.0, 475.0, 600.0, np.inf, -np.inf])

        self.run_spectrum(viewer, ["{Length}", "viridis"], io.StringIO())

        cmap, _ = spectrum.get_colormap("viridis")
        for index, position in enumerate((0.0, 0.5, 1.0)):
            np.testing.assert_allclose(viewer.current_colors[index], cmap(position))
        np.testing.assert_array_equal(
            viewer.current_colors[3:], [(0.8, 0.8, 0.8, 0.4)] * 2
        )
        self.assertIn("applied to 3 nodes", viewer.console_text.text)
        self.assertIn("(min: 350.0, max: 600.0)", viewer.console_text.text)
        self.assertIn("2 nodes with invalid values colored gray", viewer.console_text.text)

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

    def test_unknown_scheme_fails_without_changing_colors_or_saving(self):
        viewer = self.make_viewer([1.0, 3.0])
        output = TTYStringIO()
        before = viewer.current_colors.copy()

        self.run_spectrum(viewer, ["not-a-map", "{Length}"], output)

        self.assertIn("Error: Unknown color scheme 'not-a-map'", viewer.console_text.text)
        self.assertNotIn("Spectrum coloring applied", output.getvalue())
        np.testing.assert_array_equal(viewer.current_colors, before)
        viewer.promote_nodes.assert_not_called()
        viewer._save_state.assert_not_called()

    def test_missing_property_is_reported_before_an_unknown_scheme(self):
        viewer = self.make_viewer([1.0, 3.0])

        self.run_spectrum(viewer, ["Length"], io.StringIO())

        self.assertIn("Target property must be specified", viewer.console_text.text)
        self.assertNotIn("Unknown color scheme", viewer.console_text.text)
        viewer._save_state.assert_not_called()

    def test_scheme_names_are_case_insensitive_and_reported_canonically(self):
        viewer = self.make_viewer([1.0, 3.0])

        self.run_spectrum(viewer, ["{Length}", "Viridis"], io.StringIO())

        cmap, _ = spectrum.get_colormap("viridis")
        np.testing.assert_allclose(viewer.current_colors[0], cmap(0.0))
        np.testing.assert_allclose(viewer.current_colors[1], cmap(1.0))
        self.assertIn("with scheme 'viridis'", viewer.console_text.text)
        self.assertNotIn("not found", viewer.console_text.text)
        viewer._save_state.assert_called_once_with()

    def test_exact_case_wins_and_ambiguous_case_variants_are_unknown(self):
        for name, color in (("SpectrumCaseProbe", "red"), ("spectrumcaseprobe", "blue")):
            mpl.colormaps.register(mcolors.ListedColormap([color, color]), name=name)
            self.addCleanup(mpl.colormaps.unregister, name)

        viewer = self.make_viewer([1.0, 3.0])
        self.run_spectrum(viewer, ["{Length}", "spectrumcaseprobe"], io.StringIO())
        np.testing.assert_allclose(viewer.current_colors[0], mcolors.to_rgba("blue"))

        viewer = self.make_viewer([1.0, 3.0])
        self.run_spectrum(viewer, ["{Length}", "SPECTRUMCASEPROBE"], io.StringIO())
        self.assertIn(
            "Error: Unknown color scheme 'SPECTRUMCASEPROBE'", viewer.console_text.text
        )
        viewer._save_state.assert_not_called()

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

    def test_no_expression_colors_only_the_selected_visible_nodes(self):
        viewer = self.make_viewer([1.0, 2.0, 3.0, 4.0])
        viewer.selected_indices = [1, 3]
        viewer.visible_mask = np.array([True, True, True, False])

        self.run_spectrum(viewer, ["{Length}"], io.StringIO())

        promoted_mask = viewer.promote_nodes.call_args.args[0]
        np.testing.assert_array_equal(promoted_mask, [False, True, False, False])
        # The range comes from the targeted nodes, as with `$sele$`.
        self.assertIn("min: 2.0, max: 2.0", viewer.console_text.text)
        self.assertEqual(viewer.current_colors[0].tolist(), [0, 0, 0, 0])

    def test_no_expression_and_no_selection_colors_all_visible_nodes(self):
        viewer = self.make_viewer([1.0, 2.0, 3.0])
        viewer.visible_mask = np.array([True, False, True])
        output = io.StringIO()

        self.run_spectrum(viewer, ["{Length}"], output)

        promoted_mask = viewer.promote_nodes.call_args.args[0]
        np.testing.assert_array_equal(promoted_mask, [True, False, True])
        self.assertIn("Defaulting to all visible nodes", output.getvalue())
        self.assertEqual(viewer.current_colors[1].tolist(), [0, 0, 0, 0])

    def test_an_expression_overrides_the_selection(self):
        viewer = self.make_viewer([1.0, 2.0, 3.0])
        viewer.selected_indices = [0]

        self.run_spectrum(viewer, ['"node_2"', "{Length}"], io.StringIO())

        promoted_mask = viewer.promote_nodes.call_args.args[0]
        np.testing.assert_array_equal(promoted_mask, [False, False, True])

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
            (["{Length}", "viridis", "PLASMA"], "at most one color"),
            (["{Length}", "viridis", "not-a-map"], "Unrecognized extra spectrum argument"),
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


class SpectrumSelectionPrecedenceTests(unittest.TestCase):
    """A residue selection such as C53 wins over a colormap of the same name."""

    def viewer(self):
        return add_alignment(Viewer('.'))

    def test_spectrum_custom_cycle_name_cannot_override_selection(self):
        mpl.colormaps.register(mcolors.ListedColormap(['black', 'white']), name='C53')
        self.addCleanup(mpl.colormaps.unregister, 'C53')
        for scheme in ['viridis', 'coolwarm']:
            viewer = self.viewer()
            with mock.patch('commands.meta.run'):
                spectrum.run(viewer, ['C53', '{Length}', scheme])
            np.testing.assert_allclose(viewer.current_colors[0], mpl.colormaps[scheme](0.5))
            np.testing.assert_array_equal(viewer.current_colors[1:], np.ones((2, 4)))
            self.assertEqual(viewer.saved, 1)

    def test_spectrum_non_cysteine_expression_beats_colormap(self):
        mpl.colormaps.register(mcolors.ListedColormap(['black', 'white']), name='K54')
        self.addCleanup(mpl.colormaps.unregister, 'K54')
        viewer = self.viewer()
        viewer.alignment.label_to_col['54'] = 0
        with mock.patch('commands.meta.run'):
            spectrum.run(viewer, ['K54', '{Length}', 'viridis'])
        np.testing.assert_allclose(viewer.current_colors[1], mpl.colormaps['viridis'](.5))
        np.testing.assert_array_equal(viewer.current_colors[[0, 2]], np.ones((2, 4)))


class SpectrumSimilarityModeTests(unittest.TestCase):
    """One node selected and no {PROPERTY}: the nodes are colored by their score to it, from the network file."""

    def setUp(self):
        import tempfile
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        # Five network nodes; the Viewer holds the first four. Pairs: 0-1 = 10; 0-2 stored both ways
        # (20, then 30 reversed: 30 wins); 1-2 = 99 (not the reference's); 0-3 has none;
        # 0-4 = 500 reaches a node the Viewer did not load.
        self.network = self.write_network(
            [f"node_{index}" for index in range(5)],
            [(0, 1, 10.0), (0, 2, 20.0), (2, 0, 30.0), (1, 2, 99.0), (0, 4, 500.0)],
        )

    def write_network(self, headers, pairs, name="network.h5"):
        import h5py
        path = os.path.join(self.folder.name, name)
        with h5py.File(path, "w") as network:
            network.attrs["model_name"] = "BLAST"
            network.create_dataset("headers", data=[header.encode() for header in headers])
            network.create_dataset("i", data=np.array([p[0] for p in pairs], dtype=np.uint16))
            network.create_dataset("j", data=np.array([p[1] for p in pairs], dtype=np.uint16))
            network.create_dataset("score", data=np.array([p[2] for p in pairs], dtype=np.float32))
        return path

    def make_viewer(self, selected=(0,), nodes=4):
        viewer = SimpleNamespace(
            n_nodes=nodes,
            full_headers=[f"node_{index}" for index in range(nodes)],
            cluster_labels=None,
            group_labels=[set() for _ in range(nodes)],
            alignment=None,
            metadata={"Length": {"type": "number", "values": np.arange(1.0, nodes + 1)}},
            current_colors=np.zeros((nodes, 4), dtype=float),
            visible_mask=np.ones(nodes, dtype=bool),
            selected_indices=list(selected),
            console_text=SimpleNamespace(text=""),
            node_render_order=np.arange(nodes, dtype=np.int32),
            update_nodes=mock.Mock(),
            update_console_background=mock.Mock(),
            hud_displays={},
        )

        def promote_nodes(indices):
            # The real Viewer.promote_nodes: the group goes to the end of the order, sorted.
            values = np.asarray(indices)
            promoted = np.flatnonzero(values) if values.dtype == np.bool_ else np.unique(values)
            kept = viewer.node_render_order[~np.isin(viewer.node_render_order, promoted)]
            viewer.node_render_order = np.concatenate((kept, np.sort(promoted))).astype(np.int32)

        viewer.promote_nodes = mock.Mock(side_effect=promote_nodes)
        viewer.states = []
        viewer.orders = []

        def save_state():
            viewer.states.append(viewer.current_colors.copy())
            viewer.orders.append(viewer.node_render_order.copy())

        viewer._save_state = mock.Mock(side_effect=save_state)
        return viewer

    def run_spectrum(self, viewer, args, network=None):
        with mock.patch("commands.meta.run") as meta, \
                mock.patch.object(spectrum.cfg, "INPUT_HDF5", network or self.network, create=True), \
                mock.patch.object(sys, "stdout", io.StringIO()):
            spectrum.run(viewer, args)
        self.meta = meta

    def test_scores_come_from_the_file_with_the_reference_and_missing_colored_apart(self):
        viewer = self.make_viewer()

        self.run_spectrum(viewer, ["viridis"])

        cmap = mpl.colormaps["viridis"]
        _, _, reference_rgba = spectrum.reference_color(cmap)
        np.testing.assert_allclose(viewer.current_colors[1], cmap(0.0))  # score 10, the lowest
        np.testing.assert_allclose(viewer.current_colors[2], cmap(1.0))  # 30 won over 20
        np.testing.assert_allclose(viewer.current_colors[3], (0.8, 0.8, 0.8, 0.4))  # no pair: the noise color
        np.testing.assert_allclose(viewer.current_colors[0], reference_rgba)
        targets, last = (call.args[0] for call in viewer.promote_nodes.call_args_list)
        np.testing.assert_array_equal(targets, [False, True, True, True])  # the targets first
        np.testing.assert_array_equal(last, [0])  # then the reference alone
        viewer.update_nodes.assert_called_once_with()
        self.assertIn("(min: 10, max: 30)", viewer.console_text.text)
        self.assertIn("1 node(s) with no score to it", viewer.console_text.text)
        self.assertEqual(viewer.selected_indices, [0])
        self.meta.assert_not_called()

    def test_the_reference_is_drawn_last_so_it_is_always_on_top(self):
        viewer = self.make_viewer(selected=(2,))
        viewer.node_render_order = np.array([2, 0, 3, 1], dtype=np.int32)

        self.run_spectrum(viewer, ["viridis"])

        order = list(viewer.node_render_order)
        self.assertEqual(order[-1], 2)
        self.assertTrue(all(order.index(2) > order.index(target) for target in (0, 1, 3)))
        viewer._save_state.assert_called_once_with()
        np.testing.assert_array_equal(viewer.orders[0], [2, 0, 3, 1])  # undo restores this order

    def test_an_expression_leaves_the_reference_out_of_the_targets_and_on_top(self):
        viewer = self.make_viewer()

        self.run_spectrum(viewer, ["viridis", '"node_"'])

        targets = viewer.promote_nodes.call_args_list[0].args[0]
        np.testing.assert_array_equal(targets, [False, True, True, True])
        self.assertEqual(int(viewer.node_render_order[-1]), 0)

    def test_an_expression_picks_the_nodes_colored_and_sets_the_range(self):
        viewer = self.make_viewer()

        self.run_spectrum(viewer, ["viridis", '"node_2"'])

        cmap = mpl.colormaps["viridis"]
        np.testing.assert_allclose(viewer.current_colors[2], cmap(0.5))  # alone: the range is one score
        np.testing.assert_array_equal(viewer.current_colors[1], [0, 0, 0, 0])
        np.testing.assert_array_equal(viewer.current_colors[3], [0, 0, 0, 0])
        np.testing.assert_allclose(viewer.current_colors[0], spectrum.reference_color(cmap)[2])

    def test_without_an_expression_every_visible_node_is_a_target_not_the_selection(self):
        viewer = self.make_viewer()
        viewer.visible_mask = np.array([True, False, True, True])

        self.run_spectrum(viewer, ["viridis"])

        cmap = mpl.colormaps["viridis"]
        np.testing.assert_array_equal(viewer.current_colors[1], [0, 0, 0, 0])  # hidden: untouched
        np.testing.assert_allclose(viewer.current_colors[2], cmap(0.5))
        np.testing.assert_allclose(viewer.current_colors[3], spectrum.MISSING_COLOR)

    def test_a_reference_the_expression_does_not_match_still_gets_its_color(self):
        viewer = self.make_viewer()

        self.run_spectrum(viewer, ["coolwarm", '"node_1"'])

        np.testing.assert_allclose(
            viewer.current_colors[0], spectrum.reference_color(mpl.colormaps["coolwarm"])[2]
        )

    def test_the_reference_color_is_not_in_the_scheme(self):
        from matplotlib.colors import to_hex
        for scheme in ("coolwarm", "viridis"):
            cmap = mpl.colormaps[scheme]
            name, hex_code, rgba = spectrum.reference_color(cmap)
            self.assertIn(name, [candidate[0] for candidate in spectrum.REFERENCE_CANDIDATES])
            self.assertEqual(to_hex(rgba).upper(), hex_code)
            samples = np.asarray(cmap(np.linspace(0, 1, 256)))[:, :3]
            self.assertGreater(np.min(np.linalg.norm(samples - np.asarray(rgba[:3]), axis=1)), 0.3)
        # coolwarm runs blue to red through a light gray: lime or gold stand apart from it.
        self.assertIn(spectrum.reference_color(mpl.colormaps["coolwarm"])[0], ("lime", "gold"))

    def test_the_reference_color_stands_apart_from_the_canvas_background(self):
        dark = mcolors.ListedColormap(["#00E676", "#FF00FF", "#FFD600", "#00E5FF", "#000000"])  # every candidate but white
        self.assertEqual(spectrum.reference_color(dark)[0], "white")
        on_white = spectrum.reference_color(dark, (1.0, 1.0, 1.0, 1.0))
        self.assertNotEqual(on_white[0], "white")
        coolwarm = mpl.colormaps["coolwarm"]
        self.assertEqual(spectrum.reference_color(coolwarm, (1.0, 1.0, 1.0, 1.0))[0], "lime")
        self.assertEqual(spectrum.reference_color(coolwarm), spectrum.reference_color(coolwarm, None))

    def test_the_viewers_canvas_background_is_passed_on(self):
        viewer = self.make_viewer()
        viewer.canvas = SimpleNamespace(bgcolor=SimpleNamespace(rgba=np.array([1.0, 1.0, 1.0, 1.0])))
        with mock.patch.object(spectrum, "reference_color", wraps=spectrum.reference_color) as picker:
            self.run_spectrum(viewer, ["viridis"])
        np.testing.assert_array_equal(picker.call_args.args[1], [1.0, 1.0, 1.0, 1.0])

    def test_no_property_and_not_exactly_one_node_selected_fails_and_names_the_mode(self):
        for selected in ((), (0, 1)):
            viewer = self.make_viewer(selected=selected)

            self.run_spectrum(viewer, ["viridis"])

            self.assertIn("select exactly one node to color by similarity", viewer.console_text.text)
            np.testing.assert_array_equal(viewer.current_colors, np.zeros((4, 4)))
            viewer._save_state.assert_not_called()

    def test_bare_spectrum_with_one_node_selected_runs_the_similarity_mode(self):
        viewer = self.make_viewer()

        self.run_spectrum(viewer, [])

        cmap = mpl.colormaps["coolwarm"]
        name, _, rgba = spectrum.reference_color(cmap)
        self.assertEqual(name, "lime")
        np.testing.assert_allclose(viewer.current_colors[0], rgba)
        np.testing.assert_allclose(viewer.current_colors[1], cmap(0.0))
        np.testing.assert_allclose(viewer.current_colors[2], cmap(1.0))
        np.testing.assert_allclose(viewer.current_colors[3], (0.8, 0.8, 0.8, 0.4))

    def test_bare_spectrum_without_exactly_one_node_selected_fails_and_changes_nothing(self):
        for selected in ((), (0, 1)):
            viewer = self.make_viewer(selected=selected)

            self.run_spectrum(viewer, [])

            self.assertIn("select exactly one node to color by similarity", viewer.console_text.text)
            np.testing.assert_array_equal(viewer.current_colors, np.zeros((4, 4)))
            viewer._save_state.assert_not_called()

    def test_a_property_keeps_the_ordinary_mode_with_one_node_selected(self):
        viewer = self.make_viewer()

        self.run_spectrum(viewer, ["{Length}"])

        promoted = viewer.promote_nodes.call_args.args[0]
        np.testing.assert_array_equal(promoted, [True, False, False, False])  # the selection, as before
        np.testing.assert_allclose(viewer.current_colors[0], mpl.colormaps["coolwarm"](0.5))
        self.meta.assert_called_once()

    def test_a_missing_network_file_fails_and_changes_nothing(self):
        viewer = self.make_viewer()

        self.run_spectrum(viewer, ["viridis"], network=os.path.join(self.folder.name, "gone.h5"))

        self.assertIn("network file", viewer.console_text.text)
        np.testing.assert_array_equal(viewer.current_colors, np.zeros((4, 4)))
        viewer._save_state.assert_not_called()
        viewer.promote_nodes.assert_not_called()

    def test_an_unreadable_network_file_fails_and_changes_nothing(self):
        viewer = self.make_viewer()
        broken = os.path.join(self.folder.name, "broken.h5")
        with open(broken, "wb") as handle:
            handle.write(b"not an hdf5 file")

        self.run_spectrum(viewer, ["viridis"], network=broken)

        self.assertIn("could not be read", viewer.console_text.text)
        np.testing.assert_array_equal(viewer.current_colors, np.zeros((4, 4)))
        viewer._save_state.assert_not_called()

    def test_headers_that_no_longer_match_fail_and_change_nothing(self):
        viewer = self.make_viewer()
        other = self.write_network(["node_0", "node_1", "node_2"], [(0, 1, 10.0)], name="other.h5")

        self.run_spectrum(viewer, ["viridis"], network=other)

        self.assertIn("no longer matches", viewer.console_text.text)
        np.testing.assert_array_equal(viewer.current_colors, np.zeros((4, 4)))
        viewer._save_state.assert_not_called()

    def test_the_undo_state_is_saved_once_before_any_color_changes(self):
        viewer = self.make_viewer()

        self.run_spectrum(viewer, ["viridis"])

        viewer._save_state.assert_called_once_with()
        np.testing.assert_array_equal(viewer.states[0], np.zeros((4, 4)))

    def test_no_visible_target_succeeds_with_a_note_and_changes_nothing(self):
        viewer = self.make_viewer()
        viewer.visible_mask = np.array([True, False, False, False])

        self.run_spectrum(viewer, ["viridis", '"node_1"'])

        self.assertIn("No nodes matched", viewer.console_text.text)
        np.testing.assert_array_equal(viewer.current_colors, np.zeros((4, 4)))
        viewer._save_state.assert_not_called()

    def test_the_noise_color_is_shared_and_pinned(self):
        from desktop import Node_Colors
        from commands import cluster
        self.assertEqual(Node_Colors.NOISE_COLOR, (0.8, 0.8, 0.8, 0.4))
        self.assertIs(spectrum.MISSING_COLOR, Node_Colors.NOISE_COLOR)
        self.assertIs(cluster.NOISE_COLOR, Node_Colors.NOISE_COLOR)

    def test_similarity_mode_declares_its_gradient_the_missing_nodes_and_the_reference(self):
        from desktop.Node_Colors import key_state
        viewer = self.make_viewer()

        self.run_spectrum(viewer, ["viridis"])

        state = key_state(viewer)
        (layer,) = state["layers"]
        self.assertEqual((layer["kind"], layer["title"], layer["scheme"]), ("gradient", "Similarity to node_0", "viridis"))
        self.assertEqual((layer["min"], layer["max"], len(layer["stops"])), (10.0, 30.0, 16))
        self.assertEqual([entry["label"] for entry in layer["extras"]], ["No score", "Selected: node_0"])
        np.testing.assert_allclose(layer["extras"][0]["rgba"], spectrum.MISSING_COLOR)
        np.testing.assert_allclose(layer["extras"][1]["rgba"], spectrum.reference_color(mpl.colormaps["viridis"])[2])
        self.assertEqual(state["owner"].tolist(), [1, 1, 1, 1])
        self.assertEqual(state["category"].tolist(), [1, -1, -1, 0])  # the reference, two scores, one with none
        self.assertEqual(state["values"][1:3].tolist(), [10.0, 30.0])
        self.assertTrue(np.isnan(state["values"][[0, 3]]).all())
        self.assertTrue(state["labelled"].all())

    def test_property_mode_declares_the_property_gradient_and_the_nodes_it_has_no_value_for(self):
        from desktop.Node_Colors import key_state
        viewer = self.make_viewer(selected=())
        viewer.metadata["Length"]["values"] = np.array([5.0, np.nan, 15.0, 25.0])
        viewer.visible_mask = np.array([True, True, True, False])

        self.run_spectrum(viewer, ["{Length}", "plasma"])

        state = key_state(viewer)
        (layer,) = state["layers"]
        self.assertEqual((layer["title"], layer["scheme"], layer["min"], layer["max"]), ("Length", "plasma", 5.0, 15.0))
        self.assertEqual([entry["label"] for entry in layer["extras"]], ["No value"])
        self.assertEqual(state["owner"].tolist(), [1, 1, 1, -1])  # the hidden node was not painted
        self.assertEqual(state["category"].tolist(), [-1, 0, -1, -1])
        self.assertEqual(state["values"][[0, 2]].tolist(), [5.0, 15.0])
        self.assertTrue(state["labelled"][:3].all())

    def test_a_property_run_without_a_missing_value_has_no_extras(self):
        from desktop.Node_Colors import key_state
        viewer = self.make_viewer(selected=())

        self.run_spectrum(viewer, ["{Length}", "viridis"])

        self.assertEqual(key_state(viewer)["layers"][0]["extras"], [])

    def test_the_plan_is_heavy_only_without_a_property(self):
        viewer = self.make_viewer()
        self.assertTrue(spectrum.plan(viewer, ["viridis"]).heavy)
        self.assertFalse(spectrum.plan(viewer, ["{Length}"]).heavy)


if __name__ == "__main__":
    unittest.main()
