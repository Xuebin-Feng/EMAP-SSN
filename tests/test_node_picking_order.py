import os
import sys
import unittest

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from SSN_Viewer import _topmost_nearest_visible_node_index


class NodePickingOrderTests(unittest.TestCase):
    def test_identical_positions_choose_later_drawn_node(self):
        positions = np.array([[3.0, 4.0], [3.0, 4.0]], dtype=np.float32)

        selected = _topmost_nearest_visible_node_index(
            positions,
            np.array([True, True]),
            np.array([3.0, 4.0]),
        )

        self.assertEqual(selected, 1)

    def test_nearest_node_still_wins_when_distances_differ(self):
        positions = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)

        selected = _topmost_nearest_visible_node_index(
            positions,
            np.array([True, True]),
            np.array([0.25, 0.0]),
        )

        self.assertEqual(selected, 0)

    def test_identical_positions_follow_submitted_draw_order(self):
        positions = np.array(
            [[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]], dtype=np.float32
        )

        selected = _topmost_nearest_visible_node_index(
            positions,
            np.array([True, True, True]),
            np.array([3.0, 4.0]),
            np.array([2, 0, 1]),
        )

        self.assertEqual(selected, 1)

    def test_hidden_top_node_is_not_pickable(self):
        positions = np.array([[3.0, 4.0], [3.0, 4.0]], dtype=np.float32)

        selected = _topmost_nearest_visible_node_index(
            positions,
            np.array([True, False]),
            np.array([3.0, 4.0]),
        )

        self.assertEqual(selected, 0)

    def test_no_visible_nodes_returns_none(self):
        positions = np.array([[3.0, 4.0], [3.0, 4.0]], dtype=np.float32)

        selected = _topmost_nearest_visible_node_index(
            positions,
            np.array([False, False]),
            np.array([3.0, 4.0]),
        )

        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
