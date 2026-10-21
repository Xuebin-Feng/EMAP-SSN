"""Layout-cache inputs shared by the layout, settings, cache and MCP tests.

write_inputs() writes a two-sequence FASTA (set.fasta) and a one-edge alignment
network (network.h5) into a folder; settings_document() builds a layout settings
document that points at them; make_provenance() is the cache provenance a loaded
cache gives the Viewer; make_compatibility() and make_manifest() build valid
Cache_Manifest compatibility records and folder manifests from made-up hashes.
HeadlessSettingsFixture is a TestCase mixin with a temporary project root, saved
Viewer settings for those inputs, and cache generation with a stand-in layout
engine.
"""
import contextlib
import copy
import hashlib
import io
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def settings_document(temp_path, *, cache_filename="version_00.h5"):
    legacy = {
        "DIRECTORIES": {"SAVED_LAYOUT_DIR": str(temp_path / "layouts")},
        "Layout_Cache_Generator.py": {
            "NODE_FASTA_FILE": str(temp_path / "set.fasta"),
            "INPUT_HDF5": str(temp_path / "network.h5"),
            "CACHE_FILENAME": cache_filename,
            "ALIGNMENT_SCORE": "global",
            "NORM_MODE": "alignment_length",
            "SIMILARITY_THRESHOLD": 0.1,
            "TOP_EDGE_PERCENT": None,
            "UMAP_MODE": False,
            "UMAP_NEIGHBORS": 15,
            "UMAP_MIN_DIST": 0.1,
            "LAYOUT_DEVICE_SELECTION": "auto",
            "SPRING_K": 5.0,
            "COULOMB_K": 10.0,
            "COULOMB_CUTOFF": 30.0,
            "DAMPING": 0.9,
            "DT": 0.005,
            "MAX_STEPS": 10000,
            "RMSD_THRESHOLD": 0.005,
            "PERCENTAGE_DROP_THRESHOLD": 0.1,
            "RMSD_WINDOW": 50,
            "ENABLE_PROGRESSIVE_SIMULATION": False,
            "PACKING_GEOMETRY": "Square",
            "PACKING_GRID_SIZE": 20.0,
        },
    }
    from desktop.Viewer_State import DEFAULTS, encode_document
    return encode_document("layout", {**DEFAULTS, **legacy["Layout_Cache_Generator.py"],
        "SAVED_LAYOUT_DIR": str(temp_path / "layouts"), "TARGET_CACHE_PATH": None, "CACHE_NAME_MODE": "explicit"})


def write_inputs(temp_path):
    (temp_path / "set.fasta").write_text(
        ">Alpha?? Beta\nAA\n>Gamma##Delta\nCC\n", encoding="utf-8"
    )
    with h5py.File(temp_path / "network.h5", "w") as network:
        string_dtype = h5py.string_dtype("utf-8")
        network.attrs["model_name"] = "model"
        network.create_dataset(
            "headers",
            data=np.asarray(["Alpha_Beta", "Gamma_Delta"], dtype=object),
            dtype=string_dtype,
        )
        network.create_dataset("i", data=np.asarray([0], dtype=np.uint16))
        network.create_dataset("j", data=np.asarray([1], dtype=np.uint16))
        network.create_dataset("seq_lens", data=np.asarray([2, 2], dtype=np.uint16))
        for name in ("g_score", "l_score"):
            network.create_dataset(name, data=np.asarray([10], dtype=np.float32))
        for name in ("g_len", "l_len"):
            network.create_dataset(name, data=np.asarray([2], dtype=np.uint16))


_NETWORK_PREPARATION_LOADED = False


def preload_network_preparation():
    """Run both modes of network preparation once, outside any sys.modules window.

    prepare_network imports its streaming helpers lazily, and the UMAP mode's kernels
    import numba, which loads hundreds of modules on first use (and compiles or loads
    each kernel). So does the PaCMAP layout's pair module, which the generator imports
    when it lays a network out with PaCMAP. A mock.patch.dict(sys.modules, ...) window
    around a layout generation would drop everything first imported inside it on exit and
    leave orphan copies behind; run first, the window imports nothing and drops only its stub.
    """
    global _NETWORK_PREPARATION_LOADED
    if _NETWORK_PREPARATION_LOADED:
        return
    import h5py
    from desktop.Viewer_State import prepare_network
    from utilities import PaCMAP_Pairs  # noqa: F401
    with tempfile.TemporaryDirectory() as temp:
        write_inputs(pathlib.Path(temp))
        for umap_mode in (True, False):
            settings = SimpleNamespace(
                NODE_FASTA_FILE="", ALIGNMENT_SCORE="global", NORM_MODE="alignment_length",
                SIMILARITY_THRESHOLD=0.0, TOP_EDGE_PERCENT=None, UMAP_MODE=umap_mode,
                UMAP_NEIGHBORS=15,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                with h5py.File(pathlib.Path(temp) / "network.h5", "r") as network:
                    prepare_network(network, settings=settings,
                                    selected_fasta_headers=["Alpha_Beta", "Gamma_Delta"])
        # The top-percent cutoff is another helper's path.
        settings.UMAP_MODE, settings.TOP_EDGE_PERCENT = False, 50.0
        with contextlib.redirect_stdout(io.StringIO()):
            with h5py.File(pathlib.Path(temp) / "network.h5", "r") as network:
                prepare_network(network, settings=settings,
                                selected_fasta_headers=["Alpha_Beta", "Gamma_Delta"])
    _NETWORK_PREPARATION_LOADED = True


def make_provenance(manifest_id):
    """The _cache_provenance a loaded cache gives the Viewer, for one manifest id."""
    return {"cache_manifest_id": manifest_id, "layout_compatibility_json": "{}",
            "layout_compatibility_id": hashlib.sha256(b"{}").hexdigest()}


def make_compatibility(sequence_hash="a" * 64, network_hash="b" * 64, **overrides):
    """Cache_Manifest compatibility of an alignment network (threshold 0.4)."""
    import Cache_Manifest
    settings = {
        "alignment_score": "global",
        "normalization": "alignment_length",
        "umap_mode": False,
        "umap_neighbors": 15,
        "top_edge_percent": None,
        "similarity_threshold": 0.4,
    }
    settings.update(overrides)
    return Cache_Manifest.build_compatibility(
        sequence_hash,
        network_hash,
        "alignment",
        **settings,
    )


def make_manifest(compatibility, sequence_name="set.fasta", network_name="network.h5"):
    """A valid folder manifest for ``compatibility`` (input sizes are made up)."""
    import Cache_Manifest
    return Cache_Manifest.build_manifest(
        {
            "basename": sequence_name,
            "size_bytes": 10,
            "sha256": compatibility["sequence_sha256"],
        },
        {
            "basename": network_name,
            "size_bytes": 20,
            "sha256": compatibility["network_sha256"],
        },
        compatibility,
    )


class HeadlessSettingsFixture:
    """Mixin for unittest.TestCase: a temporary project root in ``self.root``.

    saved_config() writes the write_inputs() files and a viewer_settings.json
    selecting them (CPU layouts, MAX_STEPS 1, caches under custom-cache) into
    the root and returns the saved values. generate(document) generates a
    cache from a layout settings document; a stand-in engine places the two
    nodes at (0, 0) and (1, 1), so no physics, UMAP or PaCMAP code runs.
    """

    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)

    def saved_config(self):
        from desktop.Viewer_State import DEFAULTS
        write_inputs(self.root)
        values = copy.deepcopy(DEFAULTS)
        values.update(NODE_FASTA_FILE="set.fasta", INPUT_HDF5="network.h5", FASTA_DIR=str(self.root),
                      HDF5_DIR=str(self.root), CACHE_FILE_DIR=str(self.root / "custom-cache"),
                      SIMILARITY_THRESHOLD=0.1, NODE_SIZE="13", MAX_STEPS=1,
                      LAYOUT_DEVICE_SELECTION="cpu")
        (self.root / "viewer_settings.json").write_text(json.dumps(values))
        return values

    def generate(self, document):
        # The sys.modules window below drops every module first imported inside
        # it. UMAP-mode network preparation imports numba lazily (preload_network_preparation),
        # and pandas imports numpy.ma, which numpy itself does not: imported here
        # first, none of them is dropped and left behind as an orphan copy.
        import pandas  # noqa: F401
        preload_network_preparation()
        from Layout_Cache_Generator import LayoutGenerationSettings, generate_layout_cache
        engine = SimpleNamespace(calculate_layout=lambda *a: (np.array([[0, 0], [1, 1]], dtype=np.float32), 12.0))
        with mock.patch.dict(sys.modules, {"Layout_Engine_SSN": engine, "Layout_Engine_UMAP": engine,
                                          "Layout_Engine_PaCMAP": engine}):
            return generate_layout_cache(LayoutGenerationSettings.from_document(document, project_root=self.root))

    def cache_and_viewer(self, **changes):
        """Generate a cache from the saved settings with ``changes`` applied to
        the exported layout document (call saved_config() first); return the
        cache path and the exported Viewer document that selects it."""
        from desktop.Viewer_State import LAYOUT_SECTIONS
        from utilities.Headless_Settings import export_config_settings
        layout = export_config_settings("layout", self.root)["settings_document"]
        for section, fields in LAYOUT_SECTIONS.items():
            for key in fields:
                if key in changes:
                    layout[section][key] = changes[key]
        result = self.generate(layout)
        overlay = self.root / "overlay.json"
        overlay.write_text(json.dumps({"schema_version": 2, "kind": "viewer",
                                      "inputs": {"TARGET_CACHE_PATH": result.cache_path}}))
        viewer = export_config_settings("viewer", self.root, settings_path=overlay)["settings_document"]
        return result.cache_path, viewer
