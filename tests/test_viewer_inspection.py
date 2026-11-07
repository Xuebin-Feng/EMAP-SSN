"""Viewer inspection service (desktop.Viewer_Inspection).

Covers the live summary and the captured snapshots agents read: isolation
from later Viewer changes, subsets, field projection, category summaries,
cursors, long-value paging and the snapshot memory and lifetime limits.
"""
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from desktop.Viewer_Inspection import (
    ViewerInspectionService,
    ViewerInspectionError,
    SnapshotStore,
    encoded,
)
from tests.viewer_fixtures import SnapshotFixture


class ViewerInspectionTests(unittest.TestCase):
    def setUp(self):
        self.viewer = SimpleNamespace(
            n_nodes=4,
            full_headers=["A", "B", "C", "D"],
            visible_mask=[True, False, True, True],
            selected_indices=[3, 0, 3, 99],
            edges=[(0, 1), (2, 3)],
            metadata={
                "score": {"type": "numeric", "values": [1.0, float("nan"), 3.5, 4.0]},
                "family": {"type": "text", "values": ["x", "y", "x", "z"]},
            },
            cluster_labels=[1, 1, 2, 2],
            group_labels=[{"g1"}, set(), {"g2"}, {"g1", "g2"}],
            current_slider_threshold=0.75,
        )
        self.config = SimpleNamespace(
            NODE_FASTA_FILE=None,
            INPUT_HDF5="network.h5",
            MSA_FILE="",
            TARGET_CACHE_PATH=None,
            TARGET_CACHE_FILE=None,
            SIMILARITY_THRESHOLD=0.1,
        )
        self.service = ViewerInspectionService(self.viewer, self.config)

    def test_summary_is_bounded_and_json_ready(self):
        summary = self.service.get_summary()
        self.assertEqual(summary["node_count"], 4)
        self.assertEqual(summary["edge_count"], 2)
        self.assertEqual(summary["visible_node_count"], 3)
        self.assertEqual(summary["selected_node_count"], 2)
        self.assertIsNone(summary["inputs"]["node_fasta"])
        self.assertIsNone(summary["inputs"]["embedding"])
        self.assertEqual(summary["clusters"]["count"], 2)
        self.assertEqual(summary["groups"]["count"], 2)
        self.assertIsNone(summary["window_title"])

    def test_the_session_overview_gives_the_counts_and_files_the_info_card_lists(self):
        self.config.NODE_FASTA_FILE = "set.fasta"
        self.config.EMBED_FILE = "embeddings.h5"
        self.config.TARGET_CACHE_PATH = "cache/version_00.h5"
        overview = self.service.session_overview()
        self.assertEqual(
            {key: value for key, value in overview.items() if key not in ("inputs", "directories", "project_root")},
            {"node_count": 4, "visible_node_count": 3, "edge_count": 2, "shown_edge_count": 0},
        )
        # This configuration names no base folder and no project root.
        self.assertEqual((overview["directories"], overview["project_root"]), ({}, ""))
        self.assertEqual(overview["inputs"], {
            "node_fasta": "set.fasta", "network_hdf5": "network.h5", "msa": "",
            "embedding": "embeddings.h5", "layout_cache": "cache/version_00.h5",
        })
        # It agrees with the summary agents read.
        summary = self.service.get_summary()
        self.assertEqual((overview["node_count"], overview["visible_node_count"], overview["edge_count"]),
                         (summary["node_count"], summary["visible_node_count"], summary["edge_count"]))
        self.assertEqual(overview["inputs"], summary["inputs"])

    def test_the_session_overview_carries_the_base_folders_and_the_project_root_the_card_writes_aliases_from(self):
        self.config.INPUT_FILE_DIR = "C:/data/Input_Files"
        self.config.CACHE_FILE_DIR = "Cache_Files"
        self.config.EMBED_DIR = Path("D:/emb")
        self.config.ANALYSIS_RESULT_DIR = ""            # an empty setting names no folder
        self.config.PROJECT_ROOT = Path("C:/project")
        overview = self.service.session_overview()
        self.assertEqual(overview["directories"], {
            "INPUT_FILE_DIR": "C:/data/Input_Files", "CACHE_FILE_DIR": "Cache_Files", "EMBED_DIR": str(Path("D:/emb")),
        })
        self.assertEqual(overview["project_root"], str(Path("C:/project")))
        # The agents' summary does not carry them: it is the card's.
        self.assertNotIn("directories", self.service.get_summary())

    def test_the_overview_counts_a_mask_of_any_kind_and_the_edge_layers_drawn_edges(self):
        for mask, expected in (
                (np.array([True, False, True, True]), 3), ([True, False, True, True], 3),
                (np.array([True, True]), 2),            # a mask shorter than the nodes: the rest are not visible
                (np.array([False, True, True, True, True, True]), 3),  # longer: only the nodes count
                (None, 4)):
            with self.subTest(mask=mask):
                self.viewer.visible_mask = mask
                self.assertEqual(self.service.session_overview()["visible_node_count"], expected)
                self.assertEqual(self.service.get_summary()["visible_node_count"], expected)
        self.viewer.edge_layer = SimpleNamespace(drawn_edge_count=np.int64(7))
        overview = self.service.session_overview()
        self.assertEqual((overview["shown_edge_count"], type(overview["shown_edge_count"])), (7, int))
        # A Viewer with no edges has neither edges nor a layer, and no list is copied for the count.
        self.viewer.edges = None
        del self.viewer.edge_layer
        overview = self.service.session_overview()
        self.assertEqual((overview["edge_count"], overview["shown_edge_count"]), (0, 0))

    def test_the_window_title_is_reported_in_english(self):
        # The window shows its title in the chosen language; MCP clients read it in English.
        self.viewer.main_window = SimpleNamespace(windowTitle=lambda: "EMAP-SSN 查看器 [3FA9C1D2]")
        self.viewer.inspection_window_title = "EMAP-SSN Viewer [3FA9C1D2]"
        self.assertEqual(self.service.get_summary()["window_title"], "EMAP-SSN Viewer [3FA9C1D2]")
        snapshot = self.service.capture_snapshot()
        self.assertEqual(
            self.service.snapshots.execute("get_summary", snapshot)["window_title"],
            "EMAP-SSN Viewer [3FA9C1D2]",
        )


class SnapshotTests(SnapshotFixture, unittest.TestCase):
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


class SnapshotAdditionalTests(SnapshotFixture, unittest.TestCase):
    def test_a_node_pages_color_is_its_rgba_with_the_alpha_last(self):
        self.v.current_colors = np.array(
            [[1, 0, 0, 0.25], [0, 1, 0, 1], [0, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32
        )
        self.v.current_sizes = np.ones(4, dtype=np.float32)
        self.v.pos = np.zeros((4, 2), dtype=np.float32)
        sid = self.service.capture_snapshot(include_visual=True)
        rows = self.store.execute('query_nodes', sid, visual_fields=['color'])['rows']
        self.assertEqual(
            [row['visual']['color'] for row in rows],
            [[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5]],
        )

    def test_preflight_before_copy_and_foreign_snapshot(self):
        store = SnapshotStore(max_bytes=1)
        with mock.patch('desktop.Viewer_Inspection.deepcopy') as copy:
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
    def test_embedding_overview_describes_the_loaded_manifest(self):
        self.v.embedding = SimpleNamespace(
            source=object(), viewer_to_emb=np.array([1, -1, 0, -1]), embed_file='set_embeddings.h5',
            model_name='esmc_6b', width=2560, saving_mode='float16', mismatched_headers=['x'],
            load_failure=None)
        summary = self.store.execute('get_summary', self.service.capture_snapshot())
        self.assertEqual(summary['embedding'], {
            'available': True, 'file': 'set_embeddings.h5', 'model': 'esmc_6b', 'dimension': 2560,
            'saving_mode': 'float16', 'mapped_nodes': 2, 'unmapped_nodes': 2,
            'sequence_mismatches': 1, 'load_failure': None})
        self.assertEqual(summary['inputs']['embedding'], 'set_embeddings.h5')
        self.assertIn('embedding_input_v1', summary['capabilities'])
        # A file that covers no node is loaded but not available to commands.
        self.v.embedding.viewer_to_emb = np.array([-1, -1, -1, -1])
        self.assertFalse(self.store.execute('get_summary', self.service.capture_snapshot())['embedding']['available'])
        # A refused file reports why.
        self.v.embedding = SimpleNamespace(source=None, embed_file='bad.h5', load_failure='Embedding file rejected: x')
        refused = self.store.execute('get_summary', self.service.capture_snapshot())['embedding']
        self.assertFalse(refused['available'])
        self.assertEqual(refused['load_failure'], 'Embedding file rejected: x')
        self.assertEqual(refused['mapped_nodes'], 0)
    def test_every_input_a_command_can_require_has_a_summary_block(self):
        # Agents compare a command's catalog `requires` with these blocks.
        from desktop.Command_Metadata import REQUIRABLE_INPUTS
        summary = self.call('get_summary')
        for name in REQUIRABLE_INPUTS:
            with self.subTest(input=name):
                self.assertIsInstance(summary[name]['available'], bool)
    def test_empty_snapshot_and_missing_alignment(self):
        v=SimpleNamespace(n_nodes=0,full_headers=[],edges=[],metadata={},selected_indices=[],alignment=SimpleNamespace(aln=None,viewer_to_aln=[]))
        service=ViewerInspectionService(v); sid=service.capture_snapshot()
        self.assertFalse(service.snapshots.execute('get_summary',sid)['alignment']['available'])
        self.assertFalse(service.snapshots.execute('get_summary',sid)['embedding']['available'])
        self.assertEqual(service.snapshots.execute('get_summary',sid)['embedding']['file'],None)
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


class FakePartFixture:
    """A four-node Viewer whose registry declares a part that names nothing the store knows: rows of the numbers it was given."""
    def setUp(self):
        from web_ui.Plugin_Manager import WebPluginRegistry
        self.v = SimpleNamespace(n_nodes=4, full_headers=['a', 'b', 'c', 'd'], visible_mask=np.array([True, False, True, True]),
            selected_indices=[], cluster_labels=None, group_labels=None, metadata={}, edges=[], current_slider_threshold=.5)
        self.registry = self.v.web_plugin_registry = WebPluginRegistry(self.v)
        self.captured = []
        def capture(viewer, options, budget):
            self.captured.append((dict(options), budget))
            return {'summary': {'size': options.get('size', 3)}, 'values': np.arange(options.get('size', 3) * 100)}
        def rows(frozen, snapshot, indices, args):
            self.seen = (list(indices), dict(args), sorted(snapshot))
            return ({'value': int(v)} for v in frozen['values'][: int(args.get('upto', 50))])
        self.registry.register_snapshot_part('demo', 'widget', overview=lambda viewer: {'ready': True}, capture=capture,
            queries={'query_widgets': rows}, summary_arguments={'include_widget': 'switch', 'size': 'option'}, capability='widget_snapshots_v1')
        self.service = ViewerInspectionService(self.v)
        self.store = self.service.snapshots


class SnapshotPartTests(FakePartFixture, unittest.TestCase):
    def replace_part(self, **changes):
        part = self.registry._snapshot_parts['widget']
        self.registry._snapshot_parts['widget'] = type(part)(**{**vars(part), **changes})

    def test_a_snapshot_without_the_part_lists_it_without_capturing(self):
        sid = self.service.capture_snapshot()
        summary = self.store.execute('get_summary', sid)
        self.assertEqual(summary['parts'], {'widget': {'ready': True}})
        self.assertNotIn('widget_snapshots_v1', summary['capabilities'])
        self.assertEqual(self.captured, [])
        with self.assertRaisesRegex(ViewerInspectionError, 'Recapture get_summary with include_widget=true for query_widgets'):
            self.store.execute('query_widgets', sid)

    def test_a_captured_part_adds_its_summary_and_capability(self):
        sid = self.service.capture_snapshot(parts={'widget': {'size': 5}})
        summary = self.store.execute('get_summary', sid)
        self.assertEqual(summary['parts']['widget'], {'ready': True, 'captured': {'size': 5}})
        self.assertIn('widget_snapshots_v1', summary['capabilities'])
        options, budget = self.captured[0]
        self.assertEqual(options, {'size': 5})
        self.assertTrue(0 < budget <= self.store.max_bytes)

    def test_the_bytes_of_a_part_are_charged_to_the_snapshot(self):
        plain = self.service.capture_snapshot()
        with_part = self.service.capture_snapshot(parts={'widget': {'size': 5}})
        self.assertGreater(self.store.items[with_part]['bytes'] - self.store.items[plain]['bytes'], 5 * 100 * 8 - 1)
        small = SnapshotStore(max_bytes=200_000)
        with self.assertRaisesRegex(ViewerInspectionError, 'widget part exceeds'):
            small.capture(self.service, parts={'widget': {'size': 400}})
        self.assertEqual(len(small.items), 0)

    def test_a_part_that_cannot_capture_refuses_in_words_and_keeps_nothing(self):
        self.replace_part(capture=mock.Mock(side_effect=ValueError('No widget here.')))
        with self.assertRaisesRegex(ViewerInspectionError, 'No widget here.'):
            self.service.capture_snapshot(parts={'widget': {}})
        self.assertEqual(len(self.store.items), 0)

    def test_an_unknown_part_is_refused(self):
        with self.assertRaisesRegex(ViewerInspectionError, 'Unknown snapshot part: gadget'):
            self.service.capture_snapshot(parts={'gadget': {}})

    def test_a_failing_overview_is_reported_and_does_not_stop_the_snapshot(self):
        self.replace_part(overview=mock.Mock(side_effect=RuntimeError('overview broke')))
        summary = self.store.execute('get_summary', self.service.capture_snapshot())
        self.assertEqual(summary['parts']['widget'], {'error': 'overview broke'})

    def test_a_declared_query_is_answered_through_the_pager(self):
        sid = self.service.capture_snapshot(parts={'widget': {}})
        first = self.store.execute('query_widgets', sid, limit=4, upto=10)
        self.assertEqual([row['value'] for row in first['rows']], [0, 1, 2, 3])
        self.assertFalse(first['complete'])
        self.assertEqual(self.seen[0], [0, 1, 2, 3])  # no subset: every node
        self.assertEqual(self.seen[1], {'upto': 10})  # the pager's own arguments are not the query's
        self.assertIn('subsets', self.seen[2])
        second = self.store.execute('query_widgets', sid, limit=100, upto=10, cursor=first['next_cursor'])
        self.assertEqual([row['value'] for row in second['rows']], list(range(4, 10)))
        self.assertTrue(second['complete'] and second['next_cursor'] is None)
        with self.assertRaisesRegex(ViewerInspectionError, 'Invalid cursor'):
            self.store.execute('query_widgets', sid, upto=11, cursor=first['next_cursor'])
        for bad in (0, 501, 'x'):
            with self.assertRaisesRegex(ViewerInspectionError, 'limit must be 1..500'):
                self.store.execute('query_widgets', sid, limit=bad)
        for bad in (1023, 65537):
            with self.assertRaisesRegex(ViewerInspectionError, 'max_bytes must be 1024..65536'):
                self.store.execute('query_widgets', sid, max_bytes=bad)

    def test_the_byte_budget_cuts_a_page_and_a_subset_is_the_population(self):
        sid = self.service.capture_snapshot(parts={'widget': {}})
        page = self.store.execute('query_widgets', sid, limit=500, upto=300, max_bytes=1024)
        self.assertLessEqual(len(encoded(page)), 1024)
        self.assertFalse(page['complete'])
        subset = self.store.execute('create_subset', sid, scope='visible')['subset_id']
        self.store.execute('query_widgets', sid, subset_id=subset)
        self.assertEqual(self.seen[0], [0, 2, 3])

    def test_a_source_of_rows_that_can_start_late_does_not_make_the_rows_before(self):
        made = []

        class Late:
            def rows_from(self, offset):
                for position in range(offset, 20):
                    made.append(position)
                    yield {'position': position}

        self.replace_part(queries={'query_widgets': lambda *a: Late()})
        sid = self.service.capture_snapshot(parts={'widget': {}})
        first = self.store.execute('query_widgets', sid, limit=5)
        made.clear()
        second = self.store.execute('query_widgets', sid, limit=5, cursor=first['next_cursor'])
        self.assertEqual([row['position'] for row in second['rows']], [5, 6, 7, 8, 9])
        self.assertEqual(made[0], 5)

    def test_a_part_goes_with_its_snapshot_when_the_store_evicts_it(self):
        first = self.service.capture_snapshot(parts={'widget': {}})
        self.service.capture_snapshot()
        self.service.capture_snapshot()  # only two snapshots are kept: the first goes
        with self.assertRaisesRegex(ViewerInspectionError, 'expired, evicted'):
            self.store.execute('query_widgets', first)

    def test_without_a_registry_there_are_no_parts(self):
        self.v.web_plugin_registry = None
        sid = self.service.capture_snapshot()
        self.assertEqual(self.store.execute('get_summary', sid)['parts'], {})
        with self.assertRaisesRegex(ViewerInspectionError, 'Unknown snapshot part'):
            self.service.capture_snapshot(parts={'widget': {}})
        with self.assertRaisesRegex(ViewerInspectionError, 'Unknown inspection action'):
            self.store.execute('query_widgets', sid)


class RegistryPartTests(unittest.TestCase):
    def registry(self):
        from web_ui.Plugin_Manager import WebPluginRegistry
        return WebPluginRegistry(SimpleNamespace())

    def declare(self, registry, name='widget', plugin='demo', **changes):
        arguments = dict(overview=lambda viewer: {}, capture=lambda v, o, b: {'summary': {}}, queries={'query_widgets': lambda *a: []},
                         summary_arguments={'include_widget': 'switch', 'size': 'option'})
        arguments.update(changes)
        registry.register_snapshot_part(plugin, name, **arguments)

    def test_registration_is_validated(self):
        from web_ui.Plugin_Manager import WebPluginError
        for changes in ({'overview': None}, {'capture': 3}, {'queries': []}, {'queries': {'query_widgets': 5}},
                        {'queries': {'get_summary': lambda *a: []}}, {'queries': {'query_nodes': lambda *a: []}},
                        {'queries': {'Bad Name': lambda *a: []}},
                        {'summary_arguments': {'size': 'option'}}, {'summary_arguments': {'a': 'switch', 'b': 'switch'}},
                        {'summary_arguments': {'include_widget': 'switch', 'x': 'maybe'}},
                        {'summary_arguments': {'include_visual': 'switch'}}, {'capability': ' '}):
            with self.subTest(changes=changes), self.assertRaises(WebPluginError):
                self.declare(self.registry(), **changes)
        with self.assertRaises(WebPluginError):
            self.declare(self.registry(), name='Not Valid')

    def test_one_part_cannot_take_what_another_declared(self):
        from web_ui.Plugin_Manager import WebPluginError
        registry = self.registry()
        self.declare(registry)
        with self.assertRaisesRegex(WebPluginError, 'already registered by plugin'):
            self.declare(registry, plugin='other')
        with self.assertRaisesRegex(WebPluginError, 'query .* already declared'):
            self.declare(registry, name='gadget', plugin='other')
        with self.assertRaisesRegex(WebPluginError, "argument 'size' is already taken"):
            self.declare(registry, name='gadget', plugin='other', queries={'query_gadgets': lambda *a: []},
                         summary_arguments={'include_gadget': 'switch', 'size': 'option'})
        self.declare(registry)  # the same plugin again replaces its part
        self.assertEqual(list(registry.snapshot_parts()), ['widget'])
        self.assertEqual(registry.snapshot_parts()['widget'].capability, 'widget_snapshots_v1')

    def test_the_arguments_a_part_declares_are_taken_by_name(self):
        registry = self.registry()
        self.declare(registry)
        arguments = {'include_widget': True, 'size': 4, 'max_bytes': 2048}
        self.assertEqual(registry.take_part_arguments(arguments), {'widget': {'size': 4}})
        self.assertEqual(arguments, {'max_bytes': 2048})
        self.assertEqual(registry.take_part_arguments({'include_widget': False, 'size': None}), {})
        with self.assertRaisesRegex(ValueError, 'size needs include_widget=true'):
            registry.take_part_arguments({'include_widget': False, 'size': 4})
        self.assertEqual(registry.part_queries(), {'query_widgets': 'widget'})
        self.assertEqual(registry.part_capabilities(), ['widget_snapshots_v1'])

    def test_a_rolled_back_registration_takes_its_part_with_it(self):
        registry = self.registry()
        saved = registry.snapshot()
        self.declare(registry)
        registry.restore(saved)
        self.assertEqual(registry.snapshot_parts(), {})


class CentralFilesNameNoCommandTests(unittest.TestCase):
    """The user's rule: central code names no command or page; the plugin modules declare their behaviour."""
    FILES = ('src/desktop/Viewer_Inspection.py', 'src/web_ui/Web_Server.py', 'src/web_ui/Plugin_Manager.py')
    WORDS = ('include_tree', 'tree_source', 'query_clades', 'query_tree_names', 'clade')

    def words_in(self, text):
        import io
        import tokenize
        found = set()
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING, getattr(tokenize, 'FSTRING_MIDDLE', -1)):
                # Any mention in prose or a literal; a bare name "tree" is what ast code calls a syntax tree.
                lowered = token.string.lower()
                found.update(word for word in self.WORDS + ('tree',) if word in lowered)
            elif token.type == tokenize.NAME:
                found.update(word for word in self.WORDS if word in token.string.lower())
        return found

    def test_the_central_files_name_no_tree_or_clade(self):
        root = Path(__file__).resolve().parents[1]
        for name in self.FILES:
            with self.subTest(file=name):
                self.assertEqual(self.words_in((root / name).read_text(encoding='utf-8')), set())

    def test_the_guard_sees_a_word_in_a_comment_a_string_and_a_name(self):
        for text in ('x = 1  # the tree view', 'x = "include_tree"', 'query_clades = 1', 'def f():\n    """A clade."""'):
            with self.subTest(text=text):
                self.assertTrue(self.words_in(text))
        self.assertFalse(self.words_in('tree = ast.parse(text)\nfor node in ast.walk(tree): pass'))


if __name__ == '__main__':
    unittest.main()
