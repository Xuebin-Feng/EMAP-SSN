import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mcp_server.Viewer_Inspection import (
    ViewerInspectionService,
    ViewerInspectionError,
    SnapshotStore,
    encoded,
)

class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.v = SimpleNamespace(n_nodes=4, full_headers=['same', 'same', 'third', 'fourth'],
            visible_mask=np.array([True,False,True,True]), selected_indices=[],
            cluster_labels=np.array([0,0,-1,1]), group_labels=[['a','b'],['a'],[],['b']],
            metadata={'Length': {'type':'number','values':np.array([1.,2.,np.nan,np.inf])},
                      'Org,名': {'type':'text','values':np.array(['x','y','x',''],dtype=object)}}, edges=[], current_slider_threshold=.5)
        self.service=ViewerInspectionService(self.v)
        self.sid=self.service.capture_snapshot()
        self.store=self.service.snapshots
    def call(self, action, **kwargs): return self.store.execute(action,self.sid,**kwargs)
    def test_isolation_and_duplicates(self):
        self.v.metadata['Length']['values'][0]=99
        self.v.group_labels[0].append('changed')
        self.v.visible_mask[:]=False
        result=self.call('query_nodes',columns=['Length'])
        self.assertEqual(result['rows'][0]['metadata']['Length'],1)
        self.assertEqual([r['index'] for r in result['rows']],list(range(4)))
        self.assertEqual(result['rows'][0]['groups'],['a','b'])
        self.assertTrue(result['rows'][0]['visible'])
    def test_subset_expression(self):
        result=self.call('create_subset',scope='all',expression='{Length>1}&{Length<3}')
        rows=self.call('query_nodes',subset_id=result['subset_id'])['rows']
        self.assertEqual([r['index'] for r in rows],[1])
        empty=self.call('create_subset',scope='selected')
        self.assertEqual(empty['matched_count'],0)
        for expr in ('@file@','A1'):
            with self.assertRaises(ViewerInspectionError): self.call('create_subset',scope='all',expression=expr)
    def test_counts(self):
        rows=self.call('summarize_subset',columns=['Length','Org,名'])['rows']
        self.assertEqual(rows[0]['quantiles'],[1,1.25,1.5,1.75,2])
        self.assertEqual(rows[0]['missing_count'],1)
        self.assertEqual(rows[0]['invalid_count'],1)
        self.assertEqual(rows[1]['distinct_count'],2)
    def test_pages_and_binding(self):
        a=self.call('query_nodes',limit=2)
        b=self.call('query_nodes',limit=2,cursor=a['next_cursor'])
        self.assertEqual([r['index'] for r in a['rows']+b['rows']],list(range(4)))
        self.assertTrue(b['complete'])
        with self.assertRaises(ViewerInspectionError): self.call('query_nodes',columns=['Length'],cursor=a['next_cursor'])
    def test_long_value(self):
        self.v.full_headers[0]='名'*20000
        self.sid=self.service.capture_snapshot()
        result=self.call('query_nodes',max_bytes=1024)
        self.assertTrue(result['rows'][0]['node_id']['omitted'])
        self.assertLessEqual(len(encoded(result)),1024)
        chunks=[]; offset=0
        while True:
            page=self.call('read_value',index=0,field='node_id',offset=offset,max_bytes=1024)
            chunks.append(page['text']); offset=page['next_offset']
            self.assertLessEqual(len(encoded(page)),1024)
            if page['complete']: break
        self.assertEqual(json.loads(''.join(chunks)),self.v.full_headers[0])
    def test_lifetime_memory(self):
        self.service.capture_snapshot(); self.service.capture_snapshot()
        with self.assertRaisesRegex(ViewerInspectionError,'refresh'): self.call('query_nodes')
        store=SnapshotStore(max_bytes=1)
        with self.assertRaises(ViewerInspectionError): store.capture(self.service)
        clock=[0]; store=SnapshotStore(clock=lambda:clock[0]); sid=store.capture(self.service); clock[0]=901
        with self.assertRaises(ViewerInspectionError): store.execute('query_nodes',sid)


class SnapshotAdditionalTests(unittest.TestCase):
    setUp = SnapshotTests.setUp
    call = SnapshotTests.call

    def test_preflight_before_copy_and_foreign_snapshot(self):
        store = SnapshotStore(max_bytes=1)
        with mock.patch('mcp_server.Viewer_Inspection.deepcopy') as copy:
            with self.assertRaises(ViewerInspectionError):
                store.capture(self.service)
            copy.assert_not_called()
        with self.assertRaisesRegex(ViewerInspectionError, 'refresh'):
            SnapshotStore().execute('query_nodes', self.sid)

    def test_oversized_category_summaries_remain_retrievable(self):
        self.v.metadata['Org,名']['values'][0] = '名' * 10000
        self.sid = self.service.capture_snapshot()
        result = self.call('summarize_subset', columns=['Org,名'], max_bytes=1024)
        rows = result['rows'][:]
        while result['next_cursor']:
            result = self.call('summarize_subset', columns=['Org,名'], max_bytes=1024,
                               cursor=result['next_cursor'])
            self.assertLessEqual(len(encoded(result)), 1024)
            rows.extend(result['rows'])
        self.assertTrue(rows[0]['top_categories']['omitted'])
        category = next(row['category'] for row in rows if isinstance(row.get('category'), dict))
        self.assertEqual(category['read_value'], {'index': 0, 'field': 'metadata', 'column': 'Org,名'})

    def test_multiple_long_group_labels_have_unambiguous_references(self):
        self.v.group_labels[0] = {'x' * 4000, 'y' * 4000}
        self.sid = self.service.capture_snapshot()
        result = self.call('summarize_subset', max_bytes=1024)
        references = []
        while True:
            references.extend(row['category']['read_value'] for row in result['rows']
                              if isinstance(row.get('category'), dict))
            if result['complete']:
                break
            result = self.call('summarize_subset', max_bytes=1024, cursor=result['next_cursor'])
        recovered = []
        for reference in references:
            value = self.call('read_value', **reference, limit=5000)
            recovered.append(json.loads(value['text']))
        self.assertEqual(set(recovered), self.v.group_labels[0])

    def test_metadata_deletion_and_history_restore_do_not_change_snapshot(self):
        original = self.v.metadata
        self.v.metadata = {}
        without_metadata = self.service.capture_snapshot()
        # History restores the prior arrays; subsequent edits must still be isolated.
        self.v.metadata = original
        original['Length']['values'][0] = 99
        self.assertEqual(self.call('query_nodes', columns=['Length'])['rows'][0]['metadata']['Length'], 1)
        self.assertEqual(self.store.execute('describe_fields', without_metadata)['rows'], [])
    def test_refresh_after_state_changes(self):
        old=self.call('get_summary')
        self.v.metadata.clear()
        self.v.cluster_labels[:]=-1
        self.v.group_labels=[[] for _ in range(4)]
        self.v.selected_indices=[2]
        self.v.current_slider_threshold=.9
        self.v.alignment=SimpleNamespace(aln=object(),viewer_to_aln=np.array([0,-1,1,-1]),resolved_ref_full='same',msa_file='changed',offset=3)
        sid=self.service.capture_snapshot()
        new=self.store.execute('get_summary',sid)
        self.assertEqual(old['metadata_column_count'],2)
        self.assertEqual(new['metadata_column_count'],0)
        self.assertEqual(new['clusters']['noise_count'],4)
        self.assertEqual(new['clusters']['count'],0)
        self.assertEqual(new['alignment']['mapped_nodes'],2)
        self.assertEqual(new['alignment']['unmapped_nodes'],2)
        self.assertEqual(self.call('get_summary')['active_threshold'],.5)
        self.assertEqual(self.call('query_nodes')['rows'][0]['groups'],['a','b'])
    def test_empty_snapshot_and_missing_alignment(self):
        v=SimpleNamespace(n_nodes=0,full_headers=[],edges=[],metadata={},selected_indices=[],alignment=SimpleNamespace(aln=None,viewer_to_aln=[]))
        service=ViewerInspectionService(v); sid=service.capture_snapshot()
        self.assertFalse(service.snapshots.execute('get_summary',sid)['alignment']['available'])
        self.assertEqual(service.snapshots.execute('query_nodes',sid)['rows'],[])
        self.assertEqual(service.snapshots.execute('summarize_subset',sid)['rows'],[])
    def test_subset_caps_cross_snapshot_and_lru(self):
        subset=self.call('create_subset',scope='visible')['subset_id']
        second=self.service.capture_snapshot()
        with self.assertRaises(ViewerInspectionError): self.store.execute('query_nodes',second,subset_id=subset)
        for _ in range(31): self.call('create_subset',scope='selected')
        with self.assertRaises(ViewerInspectionError): self.call('create_subset',scope='all')
        self.service.capture_snapshot()
        with self.assertRaises(ViewerInspectionError): self.store.execute('query_nodes',second)
        self.call('query_nodes')
    def test_exact_nonfinite_and_group_recovery(self):
        self.assertEqual(self.call('query_nodes',columns=['Length'])['rows'][3]['metadata']['Length'],{'nonfinite':'Infinity'})
        self.v.group_labels[0]=['名'*3000,'x']
        self.sid=self.service.capture_snapshot()
        page=self.call('query_nodes',max_bytes=1024)
        self.assertTrue(page['rows'][0]['groups']['omitted'])
        text=''; offset=0
        while True:
            part=self.call('read_value',index=0,field='groups',offset=offset,max_bytes=1024)
            text+=part['text']; offset=part['next_offset']
            if part['complete']: break
        self.assertEqual(json.loads(text),self.v.group_labels[0])
    def test_many_categories_ties_and_complete_pages(self):
        self.v.metadata['Org,名']['values'][:]=['b','a','b','a']
        self.sid=self.service.capture_snapshot()
        result=self.call('summarize_subset',columns=['Org,名'],limit=1)
        self.assertEqual(result['rows'][0]['top_categories'],[('a',2),('b',2)])
        rows=result['rows']
        while result['next_cursor']:
            result=self.call('summarize_subset',columns=['Org,名'],limit=1,cursor=result['next_cursor'])
            rows+=result['rows']
        self.assertEqual([(r['category'],r['count']) for r in rows if r.get('field')=='Org,名' and 'category' in r],[('a',2),('b',2)])
    def test_selection_labels_and_unknown_references(self):
        self.v.selected_indices=[0,1]
        self.sid=self.service.capture_snapshot()
        subset=self.call('create_subset',scope='all',expression='$sele$&!{Length>1}')
        self.assertEqual(subset['matched_count'],1)
        for expr in ('{Absent=1}','#absent#','@missing@|{Length>0}'):
            with self.assertRaises(ValueError): self.call('create_subset',scope='all',expression=expr)
