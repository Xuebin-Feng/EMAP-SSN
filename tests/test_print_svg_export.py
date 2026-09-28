import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.test_print_metadata_hud import load_print_command


print_command = load_print_command()


def square_network(**overrides):
    """Four visible nodes on a square; two of the four edges pass a 0.5 threshold."""
    viewer = SimpleNamespace(
        visible_mask=np.ones(4, dtype=bool),
        pos=np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
        current_colors=np.tile([0.25, 0.5, 1.0, 1.0], (4, 1)),
        current_sizes=np.full(4, 10.0),
        current_shapes=np.array(["disc"] * 4, dtype=object),
        canvas=SimpleNamespace(bgcolor=SimpleNamespace(rgba=(1.0, 1.0, 1.0, 1.0))),
        edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32),
        edge_scores=np.array([0.90, 0.20, 0.95, 0.10]),
        current_slider_threshold=0.5,
        selected_indices=[],
    )
    for name, value in overrides.items():
        setattr(viewer, name, value)
    return viewer


def export_svg(viewer, **configuration):
    settings = {
        "EDGE_COLOR": "#000000",
        "NODE_BOUNDARY_COLOR": "#000000",
        "EDGE_ALPHA": 0.2,
        "EDGE_WIDTH": 0.5,
        "UMAP_MODE": False,
    }
    settings.update(configuration)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "network.svg")
        with mock.patch.multiple(print_command.cfg, create=True, **settings), \
                redirect_stdout(io.StringIO()):
            print_command._export_svg(viewer, path)
        with open(path, encoding="utf-8") as handle:
            return handle.read()


def elements(svg, tag):
    return [line.strip() for line in svg.splitlines() if line.strip().startswith(f"<{tag} ")]


class PrintSvgExportTests(unittest.TestCase):
    def test_svg_edges_follow_the_on_screen_similarity_threshold(self):
        viewer = square_network()

        lines = elements(export_svg(viewer), "line")

        # Scores 0.9 and 0.95 pass the 0.5 slider; 0.2 and 0.1 are hidden on screen.
        self.assertEqual(len(lines), 2)

    def test_svg_edges_match_shared_render_filter(self):
        viewer = square_network(current_slider_threshold=0.15)
        expected, _ = print_command.edge_stages(viewer, SimpleNamespace(UMAP_MODE=False))

        lines = elements(export_svg(viewer), "line")

        self.assertEqual(len(lines), len(expected))
        self.assertEqual(len(lines), 3)

    def test_svg_in_umap_mode_draws_only_edges_of_selected_nodes(self):
        viewer = square_network(current_slider_threshold=0.0, selected_indices=[0])

        lines = elements(export_svg(viewer, UMAP_MODE=True), "line")

        # Node 0 touches edges (0, 1) and (3, 0), matching the UMAP-mode display.
        self.assertEqual(len(lines), 2)

    def test_svg_uses_configured_edge_and_node_boundary_colors(self):
        svg = export_svg(
            square_network(),
            EDGE_COLOR="#ff0000",
            NODE_BOUNDARY_COLOR="#00ff00",
            EDGE_ALPHA=0.4,
        )

        lines = elements(svg, "line")
        circles = elements(svg, "circle")
        self.assertTrue(lines)
        self.assertTrue(all('stroke="rgb(255,0,0)" stroke-opacity="0.400"' in line for line in lines))
        self.assertEqual(len(circles), 4)
        self.assertTrue(all('stroke="rgb(0,255,0)"' in circle for circle in circles))


if __name__ == "__main__":
    unittest.main()
