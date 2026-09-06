# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Shared Viewer defaults, without GUI imports or personal settings."""
import os

INPUT_FILE_ALIAS = "$input_file$"
CACHE_FILE_ALIAS = "$cache_file$"
ANALYSIS_RESULT_ALIAS = "$analysis_result$"

INPUT_PROFILE_DEFAULTS = {
    "NODE_FASTA_FILE": "",
    "MSA_FILE": "",
    "INPUT_HDF5": "",
    "ALIGNMENT_SCORE": "global",
    "NORM_MODE": "alignment_length",
    "ALIGNMENT_REFERENCE": "",
    "ALIGNMENT_OFFSET": 0,
    "UMAP_MODE": False,
    "UMAP_NEIGHBORS": 15,
    "UMAP_MIN_DIST": 0.1,
    "SIMILARITY_THRESHOLD": None,
    "TOP_EDGE_PERCENT": None,
    "FILTER_MIN_OCCUPANCY": 10.0,
}


VISUAL_PROFILE_DEFAULTS = {
    "NODE_SIZE": 10,
    "EDGE_WIDTH": 1.0,
    "NODE_BOUNDARY_WIDTH": 0.5,
    "EDGE_ALPHA": 0.1,
    "TEXT_SIZE": 8,
    "TEXT_COLOR": "grey",
    "INITIAL_NODE_COLOR": "#4488ff",
    "HOVER_COLOR": "#ffaa00",
    "CONNECTED_NODE_COLOR": "#ff0000",
    "EDGE_COLOR": "#000000",
    "NODE_BOUNDARY_COLOR": "#000000",
    "LOW_RESOURCE_MODE": False,
}


PHYSICS_PROFILE_DEFAULTS = {
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
}


DIRECTORY_PROFILE_DEFAULTS = {
    "INPUT_FILE_DIR": "Input_Files",
    "CACHE_FILE_DIR": "Cache_Files",
    "ANALYSIS_RESULT_DIR": "Analysis_Results",
    "FASTA_DIR": os.path.join(INPUT_FILE_ALIAS, "Sequence_Sets"),
    "MSA_DIR": os.path.join(INPUT_FILE_ALIAS, "Multiple_Alignments"),
    "HDF5_DIR": os.path.join(INPUT_FILE_ALIAS, "Networks_EValues"),
    "METADATA_DIR": os.path.join(INPUT_FILE_ALIAS, "Meta_Data"),
    "HEADER_LIST_DIR": os.path.join(INPUT_FILE_ALIAS, "Header_Lists"),
    "SAVED_LAYOUT_DIR": os.path.join(CACHE_FILE_ALIAS, "Saved_Layouts"),
    "SETTING_EXPORT_DIR": os.path.join(CACHE_FILE_ALIAS, "Exported_Settings"),
}


LEGACY_DEFAULT_DIRECTORY_PATHS = {
    "FASTA_DIR": os.path.join("Input_Files", "Sequence_Sets"),
    "MSA_DIR": os.path.join("Input_Files", "Multiple_Alignments"),
    "HDF5_DIR": os.path.join("Input_Files", "Networks_EValues"),
    "METADATA_DIR": os.path.join("Input_Files", "Meta_Data"),
    "HEADER_LIST_DIR": os.path.join("Input_Files", "Header_Lists"),
    "SAVED_LAYOUT_DIR": os.path.join("Cache_Files", "Saved_Layouts"),
    "SETTING_EXPORT_DIR": os.path.join("Cache_Files", "Exported_Settings"),
}


PROFILE_ENUM_VALUES = {
    "ALIGNMENT_SCORE": {"global", "local"},
    "NORM_MODE": {
        "alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"
    },
    "PACKING_GEOMETRY": {"Square", "Circle"},
}


PROFILE_RANGES = {
    "ALIGNMENT_OFFSET": (-1000000, 1000000),
    "UMAP_NEIGHBORS": (2, 500),
    "UMAP_MIN_DIST": (0.0, 1.0),
    "TOP_EDGE_PERCENT": (0.0, 100.0),
    "FILTER_MIN_OCCUPANCY": (0.0, 100.0),
    "NODE_SIZE": (1, 20),
    "EDGE_WIDTH": (0.1, 3.0),
    "NODE_BOUNDARY_WIDTH": (0.0, 2.0),
    "EDGE_ALPHA": (0.0, 1.0),
    "TEXT_SIZE": (1, 24),
    "SPRING_K": (1.0, 20.0),
    "COULOMB_K": (1.0, 30.0),
    "COULOMB_CUTOFF": (1.0, 100.0),
    "DAMPING": (0.1, 2.0),
    "RMSD_WINDOW": (10, 1000),
    "PACKING_GRID_SIZE": (1.0, 200.0),
}

