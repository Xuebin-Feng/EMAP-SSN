"""Scientific semantics, projection and isolation for alignment snapshots."""
import unittest
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix
from tests.test_viewer_snapshots import SnapshotTests
from desktop.Viewer_Inspection import ViewerInspectionError, encoded


class ResidueSnapshotTests(unittest.TestCase):
    def setUp(self):
        SnapshotTests.setUp(self)
        self.v.alignment_offset = 10
        self.v.active_reference = 'ref'
        self.v.alignment = SimpleNamespace(
            aln=[SimpleNamespace(seq=s) for s in ('A-X', 'C.Z', 'AA?', 'unused')],
            viewer_to_aln=np.array([0, 1, 2, -1]),
            label_to_col={'-1': 0, '10.1': 1, '1883': 2},
            col_to_label={0:'-1',1:'10.1',2:'1883'}, resolved_ref_full='ref', msa_file='fixture.fasta')
        self.sid = self.service.capture_snapshot(include_alignment=True)

    def call(self, action, **args):
        return self.store.execute(action, self.sid, **args)

    def distribution(self, **args):
        return self.call('get_residue_distribution', positions=['-1','10.1','1883'], **args)

    def test_counts_gaps_rare_symbols_and_no_implicit_selection(self):
        self.v.selected_indices = [0]
        result = self.distribution()
        first, gap, ambiguous = result['rows']
        self.assertEqual(first['mapped_count'], 3)
        self.assertEqual(first['unmapped_count'], 1)
        self.assertEqual(first['residues'], [{'residue':'A','count':2,'fraction':2/3}, {'residue':'C','count':1,'fraction':1/3}])
        self.assertEqual(gap['gap_count'], 2)
        self.assertEqual(gap['gap_fraction'], 2/3)
        self.assertEqual({r['residue'] for r in ambiguous['residues']}, {'?','X','Z'})
        self.assertEqual(result['alignment']['offset'], 10)
        self.assertNotIn('matrix', repr(result))

    def test_sparse_equivalence_and_copy_only_network_rows(self):
        dense = self.distribution()['rows']
        codes = {1:'A',2:'C',3:'X',4:'Z',5:'?'}
        self.v.alignment.aln = SimpleNamespace(matrix=csr_matrix([[1,0,3],[2,0,4],[1,1,5],[9,9,9]]),int_to_aa=codes)
        self.sid = self.service.capture_snapshot(include_alignment=True)
        self.assertEqual(self.distribution()['rows'], dense)
        data = self.store.items[self.sid]['data']['alignment_data']
        self.assertEqual(data['matrix'].shape,(3,3))
        self.assertEqual(data['matrix'].format,'csr')
        self.v.alignment.aln.matrix.data[:] = 0
        self.assertEqual(self.distribution()['rows'], dense)

    def test_residue_subset_shared_boolean_semantics_and_isolation(self):
        subset = self.call('create_subset', scope='all', expression='!A(-1)')['subset_id']
        rows = self.call('query_nodes', subset_id=subset, fields=['index'])['rows']
        self.assertEqual(rows, [{'index':1}])  # Unmapped node is unknown, not NOT-A.
        compound = self.call('create_subset', scope='all', expression='(AC)(-1)&#a#')['subset_id']
        self.assertEqual(self.call('query_nodes', subset_id=compound, fields=['index'])['rows'],[{'index':0},{'index':1}])
        baseline = self.distribution(group_by='group')['rows']
        self.v.alignment.aln[0].seq='CCC'
        self.v.alignment.label_to_col.clear()
        self.v.alignment_offset = 999
        self.v.group_labels[0].clear()
        self.v.cluster_labels[:] = 99
        self.v.visible_mask[:] = False
        self.assertEqual(self.distribution(group_by='group')['rows'], baseline)
        self.assertEqual(self.call('create_subset', scope='visible', expression='A(-1)')['matched_count'],2)

    def test_cross_tabs_noise_overlaps_and_pages(self):
        result = self.call('get_residue_distribution',positions=['-1'],group_by='cluster')
        self.assertEqual([r['category'] for r in result['rows']], [None,-1,0,1])
        self.assertIsNone(result['rows'][-1]['gap_fraction'])
        groups = self.call('get_residue_distribution',positions=['-1'],group_by='group')
        self.assertTrue(groups['overlapping_membership'])
        self.assertEqual([r['population'] for r in groups['rows']],['overall','group','group','ungrouped'])
        self.assertEqual([r['node_count'] for r in groups['rows']], [4,2,2,1])
        page = self.call('get_residue_distribution',positions=['-1'],group_by='group',limit=1)
        collected = list(page['rows'])
        while page['next_cursor']:
            page = self.call('get_residue_distribution',positions=['-1'],group_by='group',limit=1,cursor=page['next_cursor'])
            collected += page['rows']
        self.assertEqual(collected,groups['rows'])
        first = self.distribution(limit=1)
        with self.assertRaises(ViewerInspectionError):
            self.call('get_residue_distribution',positions=['1883'],cursor=first['next_cursor'])

    def test_errors_empty_subset_and_deduplication(self):
        for positions in ([],['missing'],['1-3'],[1],['-1']*101):
            with self.assertRaises(ViewerInspectionError):
                self.call('get_residue_distribution',positions=positions)
        self.assertEqual(len(self.call('get_residue_distribution',positions=['-1','-1'])['rows']),1)
        subset=self.call('create_subset',scope='selected')['subset_id']
        row=self.call('get_residue_distribution',positions=['-1'],subset_id=subset)['rows'][0]
        self.assertEqual(row['denominator'],0)
        self.assertIsNone(row['gap_fraction'])
        for expression in ('@file@','A999'):
            count=len(self.store.items[self.sid]['subsets'])
            with self.assertRaises(ValueError): self.call('create_subset',scope='all',expression=expression)
            self.assertEqual(count,len(self.store.items[self.sid]['subsets']))
        self.sid=self.service.capture_snapshot()
        with self.assertRaisesRegex(ViewerInspectionError,'include_alignment=true'): self.distribution()
        with self.assertRaisesRegex(ViewerInspectionError,'include_alignment=true'):
            self.call('create_subset',scope='all',expression='A(-1)')

    def test_memory_limit_rejection_is_atomic(self):
        original=set(self.store.items)
        self.store.max_bytes=100
        with self.assertRaisesRegex(ValueError,'memory budget'):
            self.service.capture_snapshot(include_alignment=True)
        self.assertEqual(set(self.store.items),original)

    def test_shared_evaluator_equivalence_and_missing_memberships(self):
        import Command_Engine as ce
        mapping, valid = ce.get_alignment_mapping(self.v)
        for expr in ('A(-1)', '!A(-1)', '(AC)(-1)', 'A(-1)|#b#', '!(A(-1)&#a#)'):
            expected = ce.evaluate_selection_expression(ce.parse_selection_expression(expr), mapping, valid,
                self.v.full_headers, self.v.cluster_labels, self.v.group_labels, self.v.alignment,
                metadata=self.v.metadata, selection_mask=np.zeros(4,dtype=bool))
            subset=self.call('create_subset',scope='all',expression=expr)['subset_id']
            rows=self.call('query_nodes',subset_id=subset,fields=['index'])['rows']
            self.assertEqual([r['index'] for r in rows],np.flatnonzero(expected).tolist())
        self.v.cluster_labels=None
        self.v.group_labels=None
        self.sid=self.service.capture_snapshot(include_alignment=True)
        for mode in ('cluster','group'):
            with self.assertRaisesRegex(ViewerInspectionError,'membership data'):
                self.distribution(group_by=mode)

    def test_no_display_cutoff_and_expiry(self):
        from desktop.Alignment_Snapshot import distribution_rows
        data={'mapping':np.arange(201),'labels':{'1':0},'gaps':('-', '.'),'sparse':False,
              'sequences':('A',)*200+('Z',)}
        row=next(distribution_rows(data,range(201),['1'],'none',None,None))
        self.assertEqual(row['residues'][-1],{'residue':'Z','count':1,'fraction':1/201})
        self.store.ttl=-1
        with self.assertRaises(ViewerInspectionError): self.distribution()

    def test_missing_alignment_and_zero_coverage(self):
        self.v.alignment.viewer_to_aln[:]=-1
        self.sid=self.service.capture_snapshot(include_alignment=True)
        self.assertEqual(self.distribution()['rows'][0]['unmapped_count'],4)
        self.v.alignment=None
        with self.assertRaisesRegex(ValueError,'No alignment loaded'):
            self.service.capture_snapshot(include_alignment=True)

    def test_projection_exact_fields_size_and_cursor(self):
        full=self.call('query_nodes')
        compact=self.call('query_nodes',fields=['node_id'])
        self.assertTrue(all(set(row)=={'node_id'} for row in compact['rows']))
        self.assertLess(len(encoded(compact)),len(encoded(full)))
        first=self.call('query_nodes',fields=['node_id'],limit=1)
        with self.assertRaises(ViewerInspectionError):
            self.call('query_nodes',fields=['index'],cursor=first['next_cursor'])
        for args in ({'fields':[]},{'fields':['index','index']},{'fields':['bad']},
                     {'fields':['node_id'],'columns':['Length']},
                     {'fields':['node_id'],'visual_fields':['color']}):
            with self.assertRaises(ViewerInspectionError): self.call('query_nodes',**args)
        self.v.full_headers[0]='x'*10000
        self.sid=self.service.capture_snapshot()
        page=self.call('query_nodes',fields=['node_id'],max_bytes=1024)
        self.assertLessEqual(len(encoded(page)),1024)
        self.assertEqual(page['rows'][0]['node_id']['read_value']['index'],0)


if __name__ == '__main__': unittest.main()
