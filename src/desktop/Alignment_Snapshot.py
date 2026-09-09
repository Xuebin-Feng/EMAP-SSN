"""Frozen, network-only alignment data for read-only snapshot analysis."""
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import numpy as np


def freeze_alignment(viewer, budget):
    import Command_Engine as ce
    import EMAPSSN_Config as cfg
    alignment = getattr(viewer, 'alignment', None)
    if alignment is None or alignment.aln is None:
        raise ValueError('No alignment loaded; load an alignment before include_alignment=true.')
    mapping, _ = ce.get_alignment_mapping(viewer)
    mapping = np.asarray(mapping, dtype=int)
    rows = np.unique(mapping[mapping >= 0])
    source = alignment.aln
    sparse = hasattr(source, 'matrix')
    remapped = np.full(len(mapping), -1, dtype=int)
    remapped[mapping >= 0] = np.searchsorted(rows, mapping[mapping >= 0])
    labels = deepcopy(alignment.label_to_col)
    estimate = remapped.nbytes * 2 + sum(len(str(k)) * 4 + 128 for k in labels) + 16384
    if sparse:
        matrix = source.matrix
        # Count only retained rows, without allocating a sliced alignment first.
        if matrix.format == 'csr':
            nnz = int(np.sum(matrix.indptr[rows + 1] - matrix.indptr[rows]))
        else:
            nnz = sum(matrix.getrow(int(row)).nnz for row in rows)
        estimate += nnz * (matrix.data.dtype.itemsize + matrix.indices.dtype.itemsize) + (len(rows) + 1) * 8
    else:
        estimate += sum(len(source[int(row)].seq) * 4 + 128 for row in rows)
    if estimate > budget:
        raise ValueError('Alignment snapshot exceeds inspection memory budget.')
    data = {'mapping': remapped, 'labels': labels,
            'col_to_label': deepcopy(getattr(alignment, 'col_to_label', {})),
            'gaps': tuple(str(g).upper() for g in cfg.GAP_CHARS),
            'reference': getattr(alignment, 'resolved_ref_full', None),
            'requested_reference': getattr(viewer, 'active_reference', None),
            'offset': getattr(viewer, 'alignment_offset', 0),
            'msa': getattr(alignment, 'msa_file', None), 'sparse': sparse}
    if sparse:
        frozen = matrix[rows, :].tocsr(copy=True)
        frozen.sum_duplicates()
        frozen.sort_indices()
        data.update(matrix=frozen, codes=deepcopy(source.int_to_aa))
    else:
        data['sequences'] = tuple(str(source[int(row)].seq).upper() for row in rows)
    return data


def alignment_bytes(data):
    from desktop.Viewer_Inspection import size_of
    return sum((v.data.nbytes + v.indices.nbytes + v.indptr.nbytes + 256)
               if k == 'matrix' else size_of(v) for k, v in data.items())


def column(data, index):
    if data['sparse']:
        codes = data['matrix'][:, index].toarray().ravel()
        return np.asarray([str(data['codes'].get(int(c), 'X')).upper() if c else '-' for c in codes])
    return np.asarray([seq[index] for seq in data['sequences']])


def adapter(data):
    class Residues:
        def bulk_residue_check(self, index, residue):
            values = column(data, index)
            if residue in ('-', '.'):
                return np.isin(values, data['gaps'] + (('-',) if data['sparse'] else ()))
            return values == residue.upper()
    return SimpleNamespace(aln=Residues(), label_to_col=data['labels'], col_to_label=data['col_to_label'])


def distribution_rows(data, indices, positions, group_by, clusters, groups):
    indices = np.asarray(list(indices), dtype=int)
    def populations():
        yield 'overall', None, indices
        if group_by == 'cluster':
            for category in sorted({int(clusters[i]) for i in indices}):
                yield 'cluster', category, indices[np.asarray([clusters[i] == category for i in indices], dtype=bool)]
        elif group_by == 'group':
            for category in sorted({str(g) for i in indices for g in groups[i]}):
                yield 'group', category, np.asarray([i for i in indices if category in groups[i]], dtype=int)
            yield 'ungrouped', None, np.asarray([i for i in indices if not groups[i]], dtype=int)
    for position in positions:
        values = column(data, data['labels'][position])
        for kind, category, members in populations():
            rows = data['mapping'][members]
            mapped = rows[rows >= 0]
            counts = Counter(values[mapped].tolist())
            gap_symbols = set(data['gaps']) | ({'-'} if data['sparse'] else set())
            gaps = sum(counts.pop(symbol, 0) for symbol in gap_symbols)
            denominator = len(mapped)
            yield {'position': position, 'population': kind, 'category': category,
                   'node_count': len(members), 'mapped_count': denominator,
                   'unmapped_count': len(members) - denominator, 'denominator': denominator,
                   'gap_count': gaps, 'gap_fraction': gaps / denominator if denominator else None,
                   'residues': [{'residue': aa, 'count': count, 'fraction': count / denominator}
                                for aa, count in sorted(counts.items())]}
