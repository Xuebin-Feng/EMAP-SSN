# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

"""Canonical directory defaults shared by SSN Tools and its tool scripts."""

from __future__ import annotations

import os
from collections.abc import MutableMapping


DEFAULT_DIRECTORY_PATHS = {
    "EMBED_DIR": os.path.join("Embeddings"),
    "FASTA_DIR": os.path.join("Input_Files", "Sequence_Sets"),
    "MSA_DIR": os.path.join("Input_Files", "Multiple_Alignments"),
    "NETWORK_DIR": os.path.join("Input_Files", "Networks_EValues"),
    "REPORT_DIR": os.path.join("Analysis_Results", "Alignment_Report"),
    "SETTING_EXPORT_DIR": os.path.join("Cache_Files", "Tool_Settings"),
}


# Global directory settings consumed by each script exposed in SSN_Tools.py.
# Tool-specific paths such as SAFE_TEMP_DIR and BLASTP_DIR remain part of the
# script's own settings section rather than this shared directory registry.
TOOL_DIRECTORY_KEYS = {
    "Align_Similarity_Matrix.py": ("EMBED_DIR", "NETWORK_DIR"),
    "Align_Substitution_Matrix.py": ("FASTA_DIR", "NETWORK_DIR"),
    "Embedding_Cropping.py": ("FASTA_DIR", "EMBED_DIR"),
    "Embedding_Extraction.py": ("FASTA_DIR", "EMBED_DIR"),
    "Embedding_Injection.py": ("FASTA_DIR", "EMBED_DIR"),
    "Embedding_MSA.py": ("FASTA_DIR", "EMBED_DIR", "NETWORK_DIR", "MSA_DIR"),
    "Embedding_PWA.py": ("EMBED_DIR", "REPORT_DIR"),
    "Embedding_SSEARCH.py": ("EMBED_DIR", "REPORT_DIR"),
    "Generate_Embeddings.py": ("FASTA_DIR", "EMBED_DIR"),
    "Network_Extraction.py": ("FASTA_DIR", "NETWORK_DIR"),
    "Network_Injection.py": ("EMBED_DIR", "NETWORK_DIR"),
    "Parse_BLAST_Output.py": ("NETWORK_DIR",),
    "Sanitize_Sequences.py": ("FASTA_DIR",),
    "Sparse_MSA_Converter.py": ("MSA_DIR",),
}


def project_directory_defaults(project_root):
    """Return canonical default directories anchored to ``project_root``."""
    return {
        key: os.path.normpath(os.path.join(project_root, relative_path))
        for key, relative_path in DEFAULT_DIRECTORY_PATHS.items()
    }


def fill_missing_directory_defaults(settings):
    """Fill absent or blank global directories without replacing custom values."""
    if not isinstance(settings, MutableMapping):
        raise TypeError("Tool settings must be a mutable mapping.")

    directories = settings.get("DIRECTORIES")
    if not isinstance(directories, MutableMapping):
        directories = {}
        settings["DIRECTORIES"] = directories

    for key, default_path in DEFAULT_DIRECTORY_PATHS.items():
        value = directories.get(key)
        if value is None or not str(value).strip():
            directories[key] = default_path

    return settings
