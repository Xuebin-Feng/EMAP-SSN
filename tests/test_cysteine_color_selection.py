"""Regression coverage for Matplotlib cycle names versus residue selections."""
import unittest
from types import SimpleNamespace
from unittest import mock

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np

from tests import test_viewer_command_portal as portal_tests
from tests.test_viewer_command_portal import Viewer
from commands import color, spectrum


def add_alignment(viewer):
    labels = ['0', '1', '53', '01', '53.1', '-1', '1000']
    viewer.alignment = SimpleNamespace(
        aln=[SimpleNamespace(seq=aa * len(labels)) for aa in ['C', 'K', 'C']],
        label_to_col=dict(zip(labels, range(len(labels)))),
        viewer_to_aln=np.arange(3),
    )
    viewer.metadata = {'Length': {'type': 'number', 'values': np.array([10., 20., 30.])}}
    viewer.visible_mask[2] = False
    return viewer


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
        viewer.alignment.aln = [SimpleNamespace(seq='K' * 7) for _ in range(3)]
        viewer.selected_indices = [1]
        color.run(viewer, ['C53', 'red'])
        np.testing.assert_array_equal(viewer.current_colors, np.ones((3, 4)))
        self.assertEqual(viewer.saved, 0)

    def test_properties_still_target_selection(self):
        for value in ['red', '#123abc', 'c']:
            viewer = self.viewer()
            viewer.selected_indices = [1]
            color.run(viewer, [value, 'x0', 'square'])
            np.testing.assert_array_equal(viewer.current_colors[1], mcolors.to_rgba(value))
            self.assertEqual(viewer.current_sizes[1], 0)
            self.assertEqual(viewer.current_shapes[1], 'square')

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


class CysteinePortalTests(unittest.TestCase):
    setUpClass = classmethod(portal_tests.PortalTests.setUpClass.__func__)
    setUp = portal_tests.PortalTests.setUp
    finish = portal_tests.PortalTests.finish

    def test_cysteine_manual_portal_parity(self):
        manual = add_alignment(Viewer(self.directory.name))
        add_alignment(self.viewer)
        manual.selected_indices = self.viewer.selected_indices = [1]
        manual.process_command('color C53 red')
        result = self.finish(self.portal.submit('cysteine', 'color C53 red')['request_id'])
        self.assertEqual(result['status'], 'succeeded', result)
        np.testing.assert_array_equal(manual.current_colors, self.viewer.current_colors)
        np.testing.assert_array_equal(self.viewer.current_colors[0], mcolors.to_rgba('red'))
        np.testing.assert_array_equal(self.viewer.current_colors[1:], np.ones((2, 4)))
