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
    "PATH_DIR": os.path.join("Cache_Files", "Global_Path"),
    "REPORT_DIR": os.path.join("Cache_Files", "Align_Report"),
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
