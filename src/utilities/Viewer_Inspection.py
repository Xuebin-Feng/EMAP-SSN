# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

"""Bounded, JSON-ready, read-only inspection of an SSN Viewer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


DEFAULT_NODE_LIMIT = 100
MAX_NODE_LIMIT = 500


class ViewerInspectionError(ValueError):
    """Raised when a bounded Viewer query is invalid."""


def json_value(value):
    """Convert common scientific Python values into strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_value(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence):
        return [json_value(item) for item in value]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return json_value(item_method())
        except (TypeError, ValueError):
            pass
    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        try:
            return json_value(list_method())
        except (TypeError, ValueError):
            pass
    return str(value)


class ViewerInspectionService:
    """Produce immutable snapshots without changing Viewer state."""

    def __init__(self, viewer, configuration=None):
        self._viewer = viewer
        self._configuration = configuration

    def _node_count(self):
        viewer = self._viewer
        count = getattr(viewer, "n_nodes", None)
        if count is not None:
            return max(0, int(count))
        return len(getattr(viewer, "full_headers", ()))

    def _visible_flags(self, node_count):
        values = getattr(self._viewer, "visible_mask", None)
        if values is None:
            return [True] * node_count
        return [
            bool(values[index]) if index < len(values) else False
            for index in range(node_count)
        ]

    def _selected_indices(self, node_count):
        selected = getattr(self._viewer, "selected_indices", ()) or ()
        normalized = set()
        for value in selected:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < node_count:
                normalized.add(index)
        return sorted(normalized)

    def _metadata(self):
        metadata = getattr(self._viewer, "metadata", {})
        return metadata if isinstance(metadata, Mapping) else {}

    def _input_paths(self):
        configuration = self._configuration
        if configuration is None:
            return {}
        layout_cache = getattr(configuration, "TARGET_CACHE_PATH", None)
        if layout_cache is None:
            layout_cache = getattr(configuration, "TARGET_CACHE_FILE", None)
        return {
            "node_fasta": json_value(
                getattr(configuration, "NODE_FASTA_FILE", None)
            ),
            "network_hdf5": json_value(getattr(configuration, "INPUT_HDF5", None)),
            "msa": json_value(getattr(configuration, "MSA_FILE", None)),
            "layout_cache": json_value(layout_cache),
        }

    def get_summary(self):
        """Return a compact description of the current Viewer state."""
        viewer = self._viewer
        node_count = self._node_count()
        visible = self._visible_flags(node_count)
        selected = self._selected_indices(node_count)
        metadata = self._metadata()

        edges = getattr(viewer, "edges", ())
        try:
            edge_count = len(edges)
        except TypeError:
            edge_count = 0

        clusters = getattr(viewer, "cluster_labels", None)
        cluster_values = set()
        if clusters is not None:
            for value in clusters:
                normalized = json_value(value)
                if normalized is not None:
                    cluster_values.add(str(normalized))

        groups = getattr(viewer, "group_labels", None)
        group_names = set()
        if groups is not None:
            for memberships in groups:
                if memberships:
                    group_names.update(str(name) for name in memberships)

        threshold = getattr(viewer, "current_slider_threshold", None)
        if threshold is None and self._configuration is not None:
            threshold = getattr(self._configuration, "SIMILARITY_THRESHOLD", None)

        return {
            "inputs": self._input_paths(),
            "node_count": node_count,
            "edge_count": edge_count,
            "visible_node_count": sum(visible),
            "selected_node_count": len(selected),
            "active_threshold": json_value(threshold),
            "metadata_columns": [
                {
                    "name": str(name),
                    "type": json_value(
                        entry.get("type") if isinstance(entry, Mapping) else None
                    ),
                }
                for name, entry in metadata.items()
            ],
            "clusters": {
                "available": clusters is not None,
                "count": len(cluster_values),
            },
            "groups": {
                "available": groups is not None,
                "count": len(group_names),
            },
        }

    def query_nodes(
        self,
        scope="all",
        offset=0,
        limit=DEFAULT_NODE_LIMIT,
        columns=None,
    ):
        """Return one bounded page of nodes and requested metadata."""
        scope = str(scope).strip().lower()
        if scope not in {"all", "visible", "selected"}:
            raise ViewerInspectionError(
                "scope must be one of: all, visible, selected"
            )
        try:
            offset = int(offset)
            limit = int(limit)
        except (TypeError, ValueError) as error:
            raise ViewerInspectionError("offset and limit must be integers") from error
        if offset < 0:
            raise ViewerInspectionError("offset must be non-negative")
        if limit < 1 or limit > MAX_NODE_LIMIT:
            raise ViewerInspectionError(
                f"limit must be between 1 and {MAX_NODE_LIMIT}"
            )

        metadata = self._metadata()
        available_columns = tuple(str(name) for name in metadata)
        if columns is None:
            requested_columns = available_columns
        else:
            if isinstance(columns, str) or not isinstance(columns, Sequence):
                raise ViewerInspectionError("columns must be a list of metadata names")
            requested_columns = tuple(str(name) for name in columns)
            unknown = [name for name in requested_columns if name not in metadata]
            if unknown:
                raise ViewerInspectionError(
                    "Unknown metadata columns: " + ", ".join(unknown)
                )

        node_count = self._node_count()
        visible = self._visible_flags(node_count)
        selected = self._selected_indices(node_count)
        selected_set = set(selected)
        if scope == "visible":
            indices = [index for index in range(node_count) if visible[index]]
        elif scope == "selected":
            indices = selected
        else:
            indices = list(range(node_count))

        headers = getattr(self._viewer, "full_headers", ())
        clusters = getattr(self._viewer, "cluster_labels", None)
        groups = getattr(self._viewer, "group_labels", None)
        rows = []
        for index in indices[offset : offset + limit]:
            metadata_values = {}
            for column in requested_columns:
                entry = metadata[column]
                values = entry.get("values", ()) if isinstance(entry, Mapping) else ()
                value = values[index] if index < len(values) else None
                metadata_values[column] = json_value(value)
            rows.append(
                {
                    "index": index,
                    "node_id": (
                        json_value(headers[index])
                        if index < len(headers)
                        else str(index)
                    ),
                    "visible": visible[index],
                    "selected": index in selected_set,
                    "cluster": (
                        json_value(clusters[index])
                        if clusters is not None and index < len(clusters)
                        else None
                    ),
                    "groups": (
                        json_value(groups[index])
                        if groups is not None and index < len(groups)
                        else []
                    ),
                    "metadata": metadata_values,
                }
            )

        return {
            "scope": scope,
            "offset": offset,
            "limit": limit,
            "total": len(indices),
            "columns": list(requested_columns),
            "nodes": rows,
        }


__all__ = [
    "DEFAULT_NODE_LIMIT",
    "MAX_NODE_LIMIT",
    "ViewerInspectionError",
    "ViewerInspectionService",
    "json_value",
]
