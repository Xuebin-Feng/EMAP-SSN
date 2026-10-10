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
            viewer.current_colors[3:], [(0.7, 0.7, 0.7, 1.0)] * 2
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


if __name__ == "__main__":
    unittest.main()
