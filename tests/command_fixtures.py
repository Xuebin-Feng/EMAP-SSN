"""Fake viewers shared by the Viewer command tests (color, group, hide, ...).

one_node_viewer() is a one-node viewer whose update methods are mocks, so a test
can check what a command changed and which updates it asked for. add_alignment()
gives a viewer_fixtures.Viewer a three-row C/K/C alignment and a Length column.
reported_outcomes() records which outcome a command reported. network_viewer()
and BARBELL_EDGES give the clustering commands a small network, and
run_command() runs a command quietly and returns its outcomes and output.
"""
import io
import sys
from contextlib import contextmanager, redirect_stdout
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
    tests of a failure path check that succeeded was never called. A message
    is recorded as its English text, as the command portal records it, so a
    Message compares equal to the text it prints.
    """
    import Command_Engine

    succeeded, failed = mock.Mock(), mock.Mock()

    def in_english(record):
        def report(viewer, message=None, *args, **kwargs):
            return record(viewer, None if message is None else str(message), *args, **kwargs)
        return report

    with mock.patch.object(Command_Engine, "command_succeeded", side_effect=in_english(succeeded)), \
            mock.patch.object(Command_Engine, "command_failed", side_effect=in_english(failed)):
        yield succeeded, failed


def run_command(command, viewer, args):
    """Run a command module quietly; return (succeeded, failed, printed text)."""
    output = io.StringIO()
    with reported_outcomes() as (succeeded, failed), redirect_stdout(output):
        command.run(viewer, args)
    return succeeded, failed, output.getvalue()


# Two triangles, 0-1-2 and 3-4-5, joined by the bridge 2-3. The Jaccard index
# of an edge (neighbours shared by its endpoints / all their neighbours) is 1/3
# for 0-1 and 4-5, 1/4 for the four triangle edges that touch the bridge, and
# 0 for the bridge, whose endpoints share no neighbour.
BARBELL_EDGES = ((0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5))


def network_viewer(n_nodes, edges, edge_scores=None, cluster_labels=None):
    """A viewer holding an n_nodes network for the cluster and subcluster commands.

    edges become the (E, 2) int32 array the Viewer loads, shape (0, 2) without
    edges. edge_scores, when given, become viewer.edge_scores; cluster_labels
    make a clustered viewer whose nodes are in no group yet.
    """
    viewer = SimpleNamespace(
        n_nodes=n_nodes,
        edges=np.array(edges, dtype=np.int32).reshape(-1, 2),
        console_text=SimpleNamespace(text=""),
        current_colors=np.zeros((n_nodes, 4)),
        _save_state=mock.Mock(),
        update_nodes=mock.Mock(),
    )
    if edge_scores is not None:
        viewer.edge_scores = np.asarray(edge_scores, dtype=np.float32)
    if cluster_labels is not None:
        viewer.cluster_labels = np.asarray(cluster_labels)
        viewer.group_labels = [set() for _ in range(n_nodes)]
    return viewer


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
