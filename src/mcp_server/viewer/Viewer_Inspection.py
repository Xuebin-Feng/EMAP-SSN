# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Bounded, JSON-ready, read-only inspection of an EMAP-SSN Viewer."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import threading
import time
import uuid

import numpy as np


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


def record_value(value):
    """Strict JSON without silently conflating non-finite measurements with null."""
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return {'nonfinite': 'NaN' if math.isnan(value) else 'Infinity' if value > 0 else '-Infinity'}
    return json_value(value)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def size_of(value):
    # Conservative accounting, including object-array referents and repeated strings.
    if isinstance(value, np.ndarray):
        return sys.getsizeof(value) + value.nbytes + (sum(size_of(x) for x in value.flat) if value.dtype.hasobject else 0)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(size_of(k) + size_of(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sys.getsizeof(value) + sum(size_of(x) for x in value)
    return sys.getsizeof(value)


class SnapshotStore:
    def __init__(self, max_bytes=256 * 1024**2, ttl=900, clock=time.monotonic):
        self.max_bytes, self.ttl, self.clock = max_bytes, ttl, clock
        self.items = OrderedDict()
        self.lock = threading.RLock()

    def capture(self, service, include_visual=False):
        # Called exclusively through the owning Qt thread.
        v = service._viewer
        n = service._node_count()
        source = {'headers': getattr(v, 'full_headers', ()),
                  'metadata': {k: {'type': entry['type'], 'values': entry['values']} for k, entry in service._metadata().items()},
                  'visible': getattr(v, 'visible_mask', [True] * n),
                  'selected': service._selected_indices(n),
                  'clusters': getattr(v, 'cluster_labels', None), 'groups': getattr(v, 'group_labels', None)}
        source['inputs'] = service._input_paths()
        if include_visual:
            source['visual'] = {k: getattr(v, a, None) for k, a in {'position': 'pos', 'color': 'current_colors', 'size': 'current_sizes'}.items()}
        estimate = size_of(source) + n * 64 + 16384
        with self.lock:
            self._expire()
            if estimate > self.max_bytes:
                raise ViewerInspectionError('Snapshot exceeds inspection memory budget; reduce metadata before capturing.')
            while self.items and (len(self.items) >= 2 or sum(s['bytes'] for s in self.items.values()) + estimate > self.max_bytes):
                idle = next((key for key, item in self.items.items() if not item['active']), None)
                if idle is None:
                    raise ViewerInspectionError('Inspection snapshots are busy; retry capture when idle.')
                del self.items[idle]
            data = deepcopy(source)
            alignment = getattr(v, 'alignment', None)
            mapping = getattr(alignment, 'viewer_to_aln', ())
            data['overview'] = {'session_id': getattr(v, 'inspection_session_id', None),
                'captured_at': datetime.now(timezone.utc).isoformat(), 'node_count': n,
                'inputs': data.pop('inputs'), 'loaded_edge_count': len(getattr(v, 'edges', ())),
                'window_title': v.main_window.windowTitle() if getattr(v, 'main_window', None) is not None else None,
                'active_threshold': json_value(getattr(v, 'current_slider_threshold', getattr(service._configuration, 'SIMILARITY_THRESHOLD', None))),
                'threshold_comparison': '>= on viewer edge_scores',
                'visible_node_count': sum(bool(x) for x in data['visible']), 'selected_node_count': len(data['selected']),
                'sequence_available': bool(getattr(v, 'sequences_map', None)),
                'alignment': {'available': getattr(alignment, 'aln', None) is not None,
                              'mapped_nodes': sum(int(x) >= 0 for x in mapping),
                              'unmapped_nodes': n - sum(int(x) >= 0 for x in mapping),
                              'reference': json_value(getattr(alignment, 'resolved_ref_full', None)),
                              'requested_reference': json_value(getattr(v, 'active_reference', None)),
                              'offset': getattr(v, 'alignment_offset', 0)},
                'metadata_column_count': len(data['metadata']), 'provenance': {'status': 'unavailable'},
                'clustering_parameters': json_value(getattr(v, 'last_cluster_params', None)),
                'network_settings': {k: json_value(getattr(service._configuration, k, None)) for k in ('ALIGNMENT_SCORE', 'NORM_MODE', 'INPUT_IS_EVALUE')},
                'capabilities': ['describe_fields', 'create_subset', 'summarize_subset', 'query_nodes', 'read_value']}
            from Viewer_Visual_State import visual_overview
            data['overview'].update(visual_overview(v, service._configuration))
            data['overview']['visual_fields_available'] = list(data.get('visual', {}))
            if alignment is not None:
                data['overview']['inputs']['msa'] = json_value(getattr(alignment, 'msa_file', None))
            sid = uuid.uuid4().hex
            self.items[sid] = {'data': data, 'bytes': estimate, 'touched': self.clock(), 'subsets': {}, 'active': 0, 'lock': threading.Lock()}
            return sid

    def _expire(self):
        for key in list(self.items):
            if not self.items[key]['active'] and self.clock() - self.items[key]['touched'] >= self.ttl:
                del self.items[key]

    def execute(self, action, snapshot_id, **args):
        with self.lock:
            self._expire()
            if snapshot_id not in self.items:
                raise ViewerInspectionError('Snapshot expired, evicted, or belongs to another Viewer; refresh with get_summary.')
            s = self.items[snapshot_id]
            s['touched'] = self.clock()
            self.items.move_to_end(snapshot_id)
            s['active'] += 1
        try:
            with s['lock']:
                return self._execute(s, action, snapshot_id, **args)
        finally:
            with self.lock:
                s['active'] -= 1
                s['touched'] = self.clock()

    def _execute(self, s, action, sid, **args):
        budget = args.pop('max_bytes', 16384)
        if type(budget) is not int or not 1024 <= budget <= 65536:
            raise ViewerInspectionError('max_bytes must be 1024..65536')
        d = s['data']; n = d['overview']['node_count']
        base = {'snapshot_id': sid, 'population_size': n, 'exact': True}
        subset = args.pop('subset_id', None)
        if subset is not None and subset not in s['subsets']:
            raise ViewerInspectionError('Unknown subset for this snapshot')
        indices = s['subsets'][subset] if subset else range(n)
        base.update(subset_id=subset, matched_count=len(indices))
        if action == 'get_summary':
            result = dict(base, **d['overview'], complete=True)
            result['clusters'] = {'available': d['clusters'] is not None, 'noise_count': int(sum(x == -1 for x in d['clusters'])) if d['clusters'] is not None else 0,
                                  'count': len(set(d['clusters']) - {-1}) if d['clusters'] is not None else 0}
            result['groups'] = {'available': d['groups'] is not None, 'count': len({g for row in d['groups'] for g in row}) if d['groups'] is not None else 0}
            result['metadata_preview'] = [{'name': c, 'type': e['type']} for c, e in list(d['metadata'].items())[:5]]
            from utilities.Cache_Metadata import read_cache_metadata
            provenance = read_cache_metadata(result['inputs'].get('layout_cache'))
            result['provenance'] = {k: provenance.get(k) for k in ('status', 'cache_filename')}
            result['provenance']['cache_manifest_id'] = provenance.get('attributes', {}).get('cache_manifest_id')
            result['provenance']['generation_parameters'] = provenance.get('generation_parameters')
            result['provenance']['network_compatibility'] = (provenance.get('folder_manifest') or {}).get('compatibility')
            return self._fit(result, budget)
        if action == 'create_subset':
            scope = args.pop('scope'); expression = args.pop('expression', None)
            if scope not in ('all', 'visible', 'selected'):
                raise ViewerInspectionError('Invalid scope')
            if len(s['subsets']) >= 32:
                raise ViewerInspectionError('Snapshot subset limit reached; refresh snapshot')
            mask = np.ones(n, dtype=bool)
            if scope == 'visible': mask = np.asarray(d['visible'], dtype=bool).copy()
            if scope == 'selected':
                mask[:] = False; mask[d['selected']] = True
            if expression:
                import Command_Engine as ce
                tree = ce.parse_selection_expression(expression)
                def check(node):
                    if hasattr(node, 'kind') and node.kind not in ('string', 'metadata', 'label', 'selection'):
                        raise ViewerInspectionError('File and residue predicates are not supported by read-only metadata inspection')
                    for key in ('operand', 'left', 'right'):
                        if hasattr(node, key): check(getattr(node, key))
                check(tree)
                selected = np.zeros(n, dtype=bool); selected[d['selected']] = True
                mask &= ce.evaluate_selection_expression(tree, None, None, d['headers'], d['clusters'], d['groups'], metadata=d['metadata'], selection_mask=selected)
            members = np.flatnonzero(mask)
            with self.lock:
                charge = size_of(members) + 256
                if sum(x['bytes'] for x in self.items.values()) + charge > self.max_bytes:
                    raise ViewerInspectionError('Subset exceeds snapshot memory budget')
                key = uuid.uuid4().hex; s['subsets'][key] = members; s['bytes'] += charge
            return self._fit(dict(base, subset_id=key, matched_count=len(members), complete=True), budget)
        if action == 'read_value':
            index = args['index']; field = args['field']; column = args.get('column')
            if not 0 <= index < n: raise ViewerInspectionError('Node index out of range')
            if field == 'node_id': value = str(d['headers'][index])
            elif field == 'groups': value = json_value(d['groups'][index]) if d['groups'] is not None else []
            elif field == 'metadata' and column in d['metadata']: value = record_value(d['metadata'][column]['values'][index])
            else: raise ViewerInspectionError('Unknown value field or metadata column')
            member_index = args.get('member_index')
            if member_index is not None:
                if field != 'groups' or not 0 <= member_index < len(value):
                    raise ViewerInspectionError('Group member_index out of range or used with another field')
                value = value[member_index]
            # JSON text slices also recover oversized individual membership strings exactly.
            text = encoded(value).decode('utf-8'); offset = args.get('offset', 0)
            if offset < 0: raise ViewerInspectionError('Negative offset')
            end = min(len(text), offset + args.get('limit', 2048))
            while True:
                result = dict(base, encoding='json_text', offset=offset, next_offset=end, total_characters=len(text), text=text[offset:end], complete=end >= len(text))
                if len(encoded(result)) <= budget: return result
                end = offset + (end-offset)//2
                if end == offset: raise ViewerInspectionError('Response budget too small')
        columns = args.get('columns') or []
        if any(c not in d['metadata'] for c in columns): raise ViewerInspectionError('Unknown metadata column')
        if action == 'describe_fields':
            rows = ({'name': name, 'type': entry['type'], **self._counts(entry, range(n)), 'provenance': 'unavailable'} for name, entry in d['metadata'].items())
        elif action == 'query_nodes':
            visual_fields = args.get('visual_fields') or []
            if any(f not in d.get('visual', {}) for f in visual_fields):
                raise ViewerInspectionError('Refresh get_summary with include_visual=true before requesting visual fields')
            def records():
                selected = set(d['selected'])
                for i in indices:
                    i = int(i)
                    yield {'index': i, 'node_id': json_value(d['headers'][i]), 'visible': bool(d['visible'][i]), 'selected': i in selected,
                           'cluster': json_value(d['clusters'][i]) if d['clusters'] is not None else None,
                           'groups': json_value(d['groups'][i]) if d['groups'] is not None else [],
                           'metadata': {c: record_value(d['metadata'][c]['values'][i]) for c in columns},
                           **({'visual': {f: json_value(d['visual'][f][i]) if d['visual'][f] is not None else None for f in visual_fields}} if visual_fields else {})}
            rows = records()
        elif action == 'summarize_subset':
            rows = self._summary_rows(d, indices, columns)
        else: raise ViewerInspectionError('Unknown inspection action')
        return self._page(base, rows, action, args, budget)

    def _summary_rows(self, data, indices, columns):
        """Stream result rows; only exact counters and one column's values are temporary."""
        for column in columns:
            entry = data['metadata'][column]
            stats = self._counts(entry, indices)
            if entry['type'] == 'number':
                values = [float(entry['values'][i]) for i in indices
                          if self._classify(entry['values'][i], True) == 'valid']
                quantiles = np.quantile(values, [0, .25, .5, .75, 1]).tolist() if values else None
                yield {'kind': 'metadata_summary', 'field': column, **stats, 'quantiles': quantiles}
                continue
            counts, representatives = Counter(), {}
            for index in indices:
                value = entry['values'][index]
                if self._classify(value, False) == 'valid':
                    value = str(value)
                    counts[value] += 1
                    representatives.setdefault(value, int(index))
            ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            yield {'kind': 'metadata_summary', 'field': column, **stats,
                   'distinct_count': len(counts), 'top_categories': ordered[:10],
                   'other_count': sum(count for _, count in ordered[10:])}
            for value, count in ordered:
                yield {'kind': 'metadata_category', 'field': column, 'category': value, 'count': count,
                       'value_reference': {'index': representatives[value], 'field': 'metadata', 'column': column}}
        for field in ('clusters', 'groups'):
            values = data[field]
            if values is None:
                continue
            counts, representatives = Counter(), {}
            for index in indices:
                memberships = json_value(values[index]) if field == 'groups' else [values[index]]
                for member_index, value in enumerate(memberships):
                    key = str(value)
                    counts[key] += 1
                    representatives.setdefault(key, (int(index), member_index))
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                row = {'kind': 'membership', 'field': field, 'category': value, 'count': count,
                       'noise': field == 'clusters' and value == '-1'}
                if field == 'groups':
                    index, member_index = representatives[value]
                    row['value_reference'] = {'index': index, 'field': 'groups', 'member_index': member_index}
                yield row

    @staticmethod
    def _classify(v, numeric):
        if v is None or (isinstance(v, str) and not v.strip()): return 'missing'
        if numeric:
            try:
                x = float(v)
                return 'missing' if math.isnan(x) else 'valid' if math.isfinite(x) else 'invalid'
            except (ValueError, TypeError): return 'invalid'
        return 'valid'

    def _counts(self, entry, indices):
        counts = Counter(self._classify(entry['values'][i], entry['type'] == 'number') for i in indices)
        return {k + '_count': counts[k] for k in ('valid', 'missing', 'invalid')}

    def _fit(self, result, budget):
        if len(encoded(result)) > budget:
            raise ViewerInspectionError('Response exceeds byte budget; request fewer fields or a larger max_bytes')
        return result

    def _page(self, base, rows, action, args, budget):
        cursor = args.get('cursor'); limit = args.get('limit', 25)
        if type(limit) is not int or not 1 <= limit <= 500: raise ViewerInspectionError('limit must be 1..500')
        signature = hashlib.sha256(encoded([base['snapshot_id'], base['subset_id'], action, {k:v for k,v in args.items() if k not in ('cursor', 'limit')}])).hexdigest()[:20]
        offset = 0
        if cursor:
            try:
                sig, position = cursor.split(':'); offset = int(position)
                if sig != signature or offset < 0: raise ValueError()
            except (ValueError, AttributeError): raise ViewerInspectionError('Invalid cursor for this snapshot/query')
        result = dict(base, rows=[], returned_count=0, complete=True, next_cursor=None)
        for position, row in enumerate(rows):
            if position < offset: continue
            candidate = deepcopy(row)
            if action == 'query_nodes':
                for field in ('node_id', 'groups'):
                    if len(encoded(candidate[field])) > budget // 4: candidate[field] = {'omitted': True, 'read_value': {'index': row['index'], 'field': field}}
                for c, value in candidate['metadata'].items():
                    if len(encoded(value)) > budget // 4: candidate['metadata'][c] = {'omitted': True, 'read_value': {'index': row['index'], 'field': 'metadata', 'column': c}}
            if action == 'summarize_subset':
                if 'top_categories' in candidate and len(encoded(candidate['top_categories'])) > budget // 4:
                    candidate['top_categories'] = {'omitted': True, 'retrieve': 'Continue the category-count rows for this field.'}
                if 'value_reference' in candidate:
                    reference = candidate.pop('value_reference')
                    if len(encoded(candidate['category'])) > budget // 4:
                        candidate['category'] = {'omitted': True, 'read_value': reference}
            result['rows'].append(candidate); result['returned_count'] += 1
            result.update(complete=False, next_cursor=f'{signature}:{position+1}')
            if len(encoded(result)) > budget or result['returned_count'] > limit:
                result['rows'].pop(); result['returned_count'] -= 1
                result['next_cursor'] = f'{signature}:{position}'
                if not result['rows']: raise ViewerInspectionError('One row exceeds response budget; request fewer columns or larger max_bytes')
                return self._fit(result, budget)
        result.update(complete=True, next_cursor=None)
        return self._fit(result, budget)


class ViewerInspectionService:
    """Produce immutable snapshots without changing Viewer state."""

    def __init__(self, viewer, configuration=None):
        self._viewer = viewer
        self._configuration = configuration
        self.snapshots = SnapshotStore()

    def capture_snapshot(self, include_visual=False):
        return self.snapshots.capture(self, include_visual)

    def command_action(self, action, arguments):
        from Viewer_Command_Portal import get_portal
        from Viewer_Visual_State import capture_view
        portal = get_portal(self._viewer)
        handlers = {'execute_commands': portal.submit, 'get_command_request': portal.get,
                    'list_command_requests': portal.list_requests,
                    'read_command_output': portal.read_output,
                    'capture_view': lambda request_id=None: capture_view(self._viewer, request_id),
                    'get_command_catalog': command_catalog}
        if action not in handlers:
            raise ValueError('Unknown Viewer command portal action')
        return handlers[action](**arguments)

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
            "session_id": getattr(viewer, "inspection_session_id", None),
            "session_alias": getattr(viewer, "inspection_session_alias", None),
            "window_title": viewer.main_window.windowTitle() if getattr(viewer, "main_window", None) is not None else None,
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
    "SnapshotStore",
    "ViewerInspectionError",
    "ViewerInspectionService",
    "encoded",
    "json_value",
    "record_value",
    "size_of",
]


def command_catalog(command=None):
    """Read help text from command source without executing handlers."""
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / 'commands'
    entries = []
    for path in sorted(root.glob('*.py')):
        if path.stem.startswith('_') or (command is not None and path.stem != command):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        if not any(isinstance(n, ast.FunctionDef) and n.name == 'run' for n in tree.body):
            continue
        help_text = []
        for fn in tree.body:
            if isinstance(fn, ast.FunctionDef) and fn.name in {'print_help', 'run'}:
                for node in ast.walk(fn):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str) and ('Usage:' in node.value or 'Usage\n' in node.value):
                        help_text.append(node.value)
        entries.append({'command': path.stem,
            'writes_files': path.stem in {'export','save','print','select','meta','label','logo','run','esmfold'},
            'opens_interface': path.stem in {'agent','alignment','run','meta','esmfold','export','print','label','logo'},
            'background_work': path.stem in {'label','logo','esmfold','run'},
            'help': '\n'.join(dict.fromkeys(help_text)) if command else None})
    if command is not None and not entries:
        raise ValueError('Unknown command')
    return {'commands': entries, 'note': 'Only existing Viewer commands; effects depend on arguments. Model-generated agent calls are blocked.'}
