# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve the selected viewer cache and optional alignment reference."""

import os

import Cache_Manifest as cache_manifest


def _configured_reference_text(settings):
    value = getattr(settings, "ALIGNMENT_REFERENCE", None)
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "none" else text


def _resolve_reference_header(settings):
    reference_text = _configured_reference_text(settings)
    msa_file = getattr(settings, "MSA_FILE", None)
    if not reference_text or not msa_file or not os.path.exists(msa_file):
        return None

    try:
        reference_lower = reference_text.lower()
        if os.path.splitext(os.fspath(msa_file))[1].lower() == ".h5":
            import h5py

            with h5py.File(msa_file, "r") as handle:
                if "headers" not in handle:
                    return None
                for raw_header in handle["headers"][:]:
                    header = (
                        raw_header.decode("utf-8")
                        if isinstance(raw_header, bytes)
                        else raw_header
                    )
                    if reference_lower in header.lower():
                        return header
        else:
            with open(msa_file, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith(">") and reference_lower in line.lower():
                        return line.strip()[1:]
    except Exception as error:
        print(f"Utils Warning: Could not resolve reference header: {error}")
    return None


def resolve_selected_cache(settings):
    """Return ``(cache_path, resolved_reference_header)`` for ``settings``."""
    resolved_reference = _resolve_reference_header(settings)
    saved_layout_dir = getattr(
        settings,
        "SAVED_LAYOUT_DIR",
        os.path.join("Cache_Files", "Saved_Layouts"),
    )

    explicit_relative_path = getattr(settings, "TARGET_CACHE_PATH", None)
    if explicit_relative_path:
        return (
            cache_manifest.resolve_relative_cache_path(
                saved_layout_dir, explicit_relative_path
            ),
            resolved_reference,
        )

    fasta_file = getattr(settings, "NODE_FASTA_FILE", None) or getattr(
        settings, "SEQUENCES_FILE", ""
    )
    network_file = getattr(settings, "INPUT_HDF5", "")
    network_type = cache_manifest.validate_network_schema(network_file).network_type
    settings.INPUT_IS_EVALUE = network_type == "blast"
    canonical_name = cache_manifest.build_canonical_cache_name(
        fasta_file,
        network_file,
        network_type,
        alignment_score=getattr(settings, "ALIGNMENT_SCORE", None),
        normalization=getattr(settings, "NORM_MODE", None),
        umap_mode=getattr(settings, "UMAP_MODE", False),
        umap_neighbors=getattr(settings, "UMAP_NEIGHBORS", 15),
        top_edge_percent=getattr(settings, "TOP_EDGE_PERCENT", None),
        similarity_threshold=getattr(settings, "SIMILARITY_THRESHOLD", None),
    )
    target_folder = os.path.join(saved_layout_dir, canonical_name)

    selected_cache = getattr(settings, "TARGET_CACHE_FILE", None)
    if (
        isinstance(selected_cache, str)
        and selected_cache.strip()
        and selected_cache != "None"
    ):
        cache_manifest.validate_cache_filename(selected_cache)
        return os.path.join(target_folder, selected_cache), resolved_reference

    return os.path.join(target_folder, "version_00.h5"), resolved_reference


__all__ = ["resolve_selected_cache"]
