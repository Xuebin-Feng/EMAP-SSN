"""Fake viewers shared by the Viewer command tests (color, group, hide, ...).

one_node_viewer() is a one-node viewer whose update methods are mocks, so a test
can check what a command changed and which updates it asked for. add_alignment()
gives a viewer_fixtures.Viewer a three-row C/K/C alignment and a Length column.
reported_outcomes() records which outcome a command reported.
"""
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tests.sparse_alignment import sparse_alignment  # noqa: E402


def one_node_viewer():
    """A viewer with one visible node, "node", in cluster 1 and in no group."""
    return SimpleNamespace(
        n_nodes=1,
        full_headers=["node"],
        cluster_labels=np.array([1]),
        group_labels=[set()],
        alignment=None,
        metadata={
            "Length": {
                "type": "number",
                "values": np.array([100.0]),
            }
        },
        current_colors=np.array([[0.1, 0.2, 0.3, 1.0]]),
        current_sizes=np.array([5.0]),
        visible_mask=np.array([True]),
        selected_indices=[],
        console_text=SimpleNamespace(text=""),
        _save_state=mock.Mock(),
        promote_nodes=mock.Mock(),
        update_nodes=mock.Mock(),
        update_selection_visual=mock.Mock(),
        update_edges=mock.Mock(),
    )


@contextmanager
def reported_outcomes():
    """Replace Command_Engine.command_succeeded and command_failed with mocks.

    Yields (succeeded, failed). A failure must not report success as well, so
    tests of a failure path check that succeeded was never called.
    """
    import Command_Engine

    with mock.patch.object(Command_Engine, "command_succeeded") as succeeded, \
            mock.patch.object(Command_Engine, "command_failed") as failed:
        yield succeeded, failed


def add_alignment(viewer):
    labels = ['0', '1', '53', '01', '53.1', '-1', '1000']
    viewer.alignment = SimpleNamespace(
        aln=sparse_alignment(
            (f'row{index}', aa * len(labels)) for index, aa in enumerate(['C', 'K', 'C'])
        ),
        label_to_col=dict(zip(labels, range(len(labels)))),
        viewer_to_aln=np.arange(3),
    )
    viewer.metadata = {'Length': {'type': 'number', 'values': np.array([10., 20., 30.])}}
    viewer.visible_mask[2] = False
    return viewer
