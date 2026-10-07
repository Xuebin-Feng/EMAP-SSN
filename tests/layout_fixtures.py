"""Layout-cache inputs shared by the layout, settings, cache and MCP tests.

write_inputs() writes a two-sequence FASTA (set.fasta) and a one-edge alignment
network (network.h5) into a folder; settings_document() builds a layout settings
document that points at them.
"""
import pathlib
import sys

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
