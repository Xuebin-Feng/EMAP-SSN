# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Shared two-node Viewer fixture and launch helpers for MCP Viewer tests.

Not collected by unittest; import with ``from tests.mcp_viewer_fixtures import ...``.
"""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Layout_Cache_Generator import LayoutGenerationSettings, generate_layout_cache  # noqa: E402
from desktop.Viewer_State import DEFAULTS, encode_document  # noqa: E402
from mcp_server.viewer.Viewer_Client import MCPViewerError  # noqa: E402
from tests import layout_fixtures as fixtures  # noqa: E402

# Launched Viewers must cope with whatever encoding their pipes and log files
# get: a UTF-8 override, set here or inherited from the shell, once hid a crash
# on non-UTF-8 output. Launches therefore run without one.
_ENCODING_OVERRIDES = ("PYTHONIOENCODING", "PYTHONUTF8")


def launch_environment(**values):
    """os.environ without UTF-8 encoding overrides, plus ``values``."""
    environment = {name: value for name, value in os.environ.items() if name not in _ENCODING_OVERRIDES}
    environment.update(values)
    return environment


def stop_recorded_launches(sessions_root):
    """Stop every launch process still recorded under ``sessions_root`` (test cleanup).

    A normal launch records two: the session launcher (process.json) and the
    Viewer its monitor runs (window-process.json). The monitor is detached, so
    stopping the launcher after it has ended leaves the Viewer.
    """
    from mcp_server.viewer import Viewer_Client

    for name in ("process.json", "window-process.json"):
        for identity in Path(sessions_root).glob(f"launches/*/{name}"):
            process = Viewer_Client._recorded_process(identity)
            if process is not None:
                try:
                    Viewer_Client._terminate_tree(process)
                except MCPViewerError:
                    pass


def create_viewer_fixture(test_case):
    """Build a two-node cached Viewer fixture in a temporary folder cleaned up with ``test_case``."""
    temp = tempfile.TemporaryDirectory()
    test_case.addCleanup(temp.cleanup)
    root = Path(temp.name)
    fixtures.write_inputs(root)
    settings = LayoutGenerationSettings.from_document(
        fixtures.settings_document(root), project_root=fixtures.ROOT
    )
    with mock.patch(
        "Layout_Engine_SSN.calculate_layout",
        return_value=(np.array([[0, 0], [1, 1]], dtype=np.float32), 10.0),
    ):
        cache = generate_layout_cache(settings).cache_path
    document = dict(
        TARGET_CACHE_PATH=cache,
        NODE_FASTA_FILE=str(root / "set.fasta"),
        INPUT_HDF5=str(root / "network.h5"),
        MSA_FILE="",
        UMAP_MODE=False,
        ALIGNMENT_SCORE="global",
        NORM_MODE="alignment_length",
        SIMILARITY_THRESHOLD=0.1,
        TOP_EDGE_PERCENT=None,
        # Launched Viewers create this folder; keep it out of the project.
        SAVED_LAYOUT_DIR=str(root / "layouts"),
    )
    document = encode_document("viewer", {**DEFAULTS, **document})
    return SimpleNamespace(root=root, cache=cache, document=document)
