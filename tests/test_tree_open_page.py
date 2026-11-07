# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tree View's opening of a tree (tree.html, tree_open.js, and the reader of tree_read*.js): a file chosen, dropped or pasted.

The page reads the text with the reader it copied from Cladewise (Newick and its dialects, NEXUS, PhyloXML, NeXML, JSON), asks
which tree of a file of several, sends the tree to the Viewer (the tree_upload action's route), and draws what comes back as it
draws any tree: its leaves are the network's nodes, so the colours, the selection and the hidden nodes follow. The card of how
the leaves matched the network follows each opening. The page is opened offscreen over the real web server; the fixtures are
small files of every format family (tests/tree_formats, from Cladewise's own).
"""

import json
import os
import pathlib
import sys
import time
import unittest
from unittest import mock

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for folder in (SRC_DIR, PROJECT_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests.tree_fixtures import give_edges, headers_of, node_colors  # noqa: E402
from tests.tree_page_harness import TreePageMixin  # noqa: E402
from web_ui import tree_backend  # noqa: E402

FORMATS = PROJECT_ROOT / "tests" / "tree_formats"
PRIMATES = ["Homo_sapiens", "Pan_troglodytes", "Mus_musculus", "Rattus_norvegicus", "Bos_taurus"]
GREEK = ["Alpha", "Beta", "Gamma", "Delta"]
# The network: the names of every fixture, then names no fixture has.
NETWORK = PRIMATES + GREEK + ["Epsilon", "Isolate_A", "Isolate_B", "Isolate_C", "A", "B", "C", "D"] + [f"extra_{k}" for k in range(6)]


def text_of(name):
    return (FORMATS / name).read_text(encoding="utf-8")


class OpenCase(TreePageMixin, unittest.TestCase):
    """A page over a network of ``NETWORK`` (a chain of edges), with no guide tree, so that it opens on the threshold tree."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(tree_backend._THRESHOLD_CACHE, {"order": None, "count": 0, "built": None})
        patcher.start()
        self.addCleanup(patcher.stop)

    def start(self, *, headers=None, colors=False, hidden=(), select=(), guide=False):
        headers = list(NETWORK if headers is None else headers)
        options = {} if guide else {"tree": None}
        self.serve_headers(headers, colors=node_colors(len(headers), {1: "#ff0000", 3: "#0000ff"}) if colors else None,
                           hidden=hidden, **options)
        give_edges(self.viewer, [(n, n + 1) for n in range(len(headers) - 1)], np.arange(len(headers) - 1, 0, -1), threshold=3.5)
        self.enable_selection()
        if select:
            self.viewer.select(select)
        self.open_page()
        if colors:
            self.wait_for("TreeView.state.colors !== null")
        deadline = time.monotonic() + 10
        while not self.server.has_event_client("tree") and time.monotonic() < deadline:
            self.pump(0.05)

    # ---- The three ways of opening --------------------------------------------------------------

    def open_text(self, text, name="tree.nwk"):
        self.script(f"TreeView.openText({json.dumps(text)}, {{name: {json.dumps(name)}, size: {len(text)}, pasted: false}}); return 1")

    def drop(self, text, name="tree.nwk"):
        return self.script(f"""const dt = new DataTransfer(); dt.items.add(new File([{json.dumps(text)}], {json.dumps(name)}));
            document.dispatchEvent(new DragEvent('dragenter', {{dataTransfer: dt, bubbles: true, cancelable: true}}));
            const dropping = document.body.classList.contains('dropping');
            document.dispatchEvent(new DragEvent('drop', {{dataTransfer: dt, bubbles: true, cancelable: true}}));
            return {{dropping, after: document.body.classList.contains('dropping')}}""")

    def paste(self, text, target="document.body"):
        return self.script(f"""const dt = new DataTransfer(); dt.setData('text/plain', {json.dumps(text)});
            const event = new ClipboardEvent('paste', {{clipboardData: dt, bubbles: true, cancelable: true}});
            {target}.dispatchEvent(event);
            return event.defaultPrevented""")

    def choose_file(self, text, name="tree.nwk"):
        """What the Tree list's "Open a tree file..." entry does: the file chooser opens, and a file chosen in it is opened."""
        return self.script(f"""window.__chooser = 0;
            const input = document.getElementById('file-input');
            input.click = () => {{ window.__chooser++; }};
            const select = document.getElementById('source-select');
            select.value = 'open'; select.dispatchEvent(new Event('change'));
            const dt = new DataTransfer(); dt.items.add(new File([{json.dumps(text)}], {json.dumps(name)}));
            input.files = dt.files; input.dispatchEvent(new Event('change'));
            return {{chooser: window.__chooser, selected: select.value}}""")

    def wait_uploaded(self, view="tree"):
        self.wait_for(f"TreeView.state.source === 'uploaded' && document.body.dataset.view === {json.dumps(view)}")

    def opened(self):
        return self.state("""(() => { const s = TreeView.state, m = s.model;
            return {labels: Array.from(m.labels), matched: m.match.matched, exact: m.match.exact, sanitized: m.match.sanitized,
                    unmatched: m.match.unmatched, notInTree: m.match.network_nodes_not_in_tree, format: m.uploaded.format,
                    file: m.uploaded.file, treeName: m.uploaded.tree_name, dialects: m.uploaded.dialects, rooted: m.rooted,
                    rootStated: m.uploaded.root_stated, hasLengths: m.hasLengths, support: TreeView.hasSupport(m),
                    supportName: m.uploaded.support_name, cladeNames: m.uploaded.clade_names,
                    warnings: m.warnings.map(w => w.code), view: document.body.dataset.view}; })()""")

    def shot(self, name):
        """Save a picture of the page for the person reviewing the work (the folder is the scratchpad's)."""
        folder = os.environ.get("TREE_SHOTS_DIR")
        if not folder or not pathlib.Path(folder).is_dir():
            return
        self.pump(0.3)
        self.view.grab().save(str(pathlib.Path(folder) / name))


class FormatTests(OpenCase):
    """A fixture of every format family, opened as text, is a tree whose leaves the network has."""

    CASES = [
        # file, leaves, format, dialects, has support, leaves matched, of them by the header rules
        ("primates.nwk", PRIMATES, "Newick", [], True, 5, 0),
        ("iqtree.treefile", PRIMATES, "Newick", None, True, 5, 0),
        ("raxml_support.nwk", PRIMATES, "Newick", None, True, 5, 0),
        ("nhx.nwk", PRIMATES, "Newick", None, True, 5, 0),
        ("beast.tre", GREEK + ["Epsilon"], "NEXUS", None, True, 5, 0),
        ("mrbayes.con.tre", GREEK, "NEXUS", None, True, 4, 0),
        ("figtree.nex", GREEK, "NEXUS", None, True, 4, 0),
        ("single.nex", GREEK, "NEXUS", [], True, 4, 0),
        ("primates.phyloxml.xml", ["Homo sapiens", "Pan_troglodytes", "Mus musculus & relatives", "Rattus_norvegicus", "BOVTA"],
         "PhyloXML", None, True, 3, 1),
        ("primates.nexml.xml", PRIMATES, "NeXML", None, None, 5, 0),
        ("nested.json", GREEK, "JSON (nested)", None, True, 4, 0),
        ("auspice.json", ["Isolate_A", "Isolate_B", "Isolate_C"], "JSON (Auspice)", None, True, 3, 0),
        ("placements.jplace", PRIMATES, None, None, None, 5, 0),
    ]

    def test_every_format_family_opens_and_matches_the_network(self):
        self.start()
        for name, leaves, file_format, dialects, support, matched, rules in self.CASES:
            with self.subTest(name):
                self.open_text(text_of(name), name)
                self.wait_for(f"(TreeView.state.source === 'uploaded' && TreeView.state.model && TreeView.state.model.uploaded.file === {json.dumps(name)})"
                              " || (document.body.dataset.view === 'empty' && !document.getElementById('empty-choice').hidden)")
                if self.state("document.body.dataset.view === 'empty'"):
                    # PhyloXML and NeXML files of these fixtures hold two trees: the first is the one with the primates.
                    self.script("document.getElementById('empty-choice').selectedIndex = 0; document.getElementById('empty-open').click(); return 1")
                    self.wait_for(f"TreeView.state.source === 'uploaded' && TreeView.state.model && TreeView.state.model.uploaded.file === {json.dumps(name)}")
                found = self.opened()
                self.assertEqual(found["view"], "tree")
                self.assertEqual(sorted(found["labels"]), sorted(leaves))
                if file_format is not None:
                    self.assertEqual(found["format"], file_format)
                self.assertEqual(found["file"], name)
                self.assertEqual((found["matched"], found["unmatched"], found["sanitized"]), (matched, len(leaves) - matched, rules))
                if dialects is not None:
                    self.assertEqual(found["dialects"], dialects)
                if support is not None:
                    self.assertEqual(found["support"], support)
        self.assertEqual(self.console_errors(), [])

    def test_the_names_of_phyloxml_with_spaces_match_by_the_headers_rules(self):
        self.start()
        self.open_text(text_of("primates.phyloxml.xml"), "primates.phyloxml.xml")
        self.wait_for("document.body.dataset.view === 'empty' && !document.getElementById('empty-choice').hidden")
        self.script("document.getElementById('empty-choice').selectedIndex = 0; document.getElementById('empty-open').click(); return 1")
        self.wait_uploaded()
        found = self.opened()
        # "Homo sapiens" is the network's Homo_sapiens only under the Viewer's own header rules.
        self.assertIn("Homo sapiens", found["labels"])
        self.assertEqual(found["sanitized"], 1)
        self.assertEqual(found["exact"], found["matched"] - 1)

    def test_what_the_file_says_of_the_tree_is_kept(self):
        self.start()
        self.open_text(text_of("single.nex"), "single.nex")
        self.wait_uploaded()
        found = self.opened()
        self.assertEqual((found["treeName"], found["rooted"], found["rootStated"], found["hasLengths"]), ("one", True, True, True))
        self.assertEqual(found["cladeNames"], 1)
        self.open_text(text_of("mrbayes.con.tre"), "mrbayes.con.tre")
        self.wait_for("TreeView.state.model.uploaded.file === 'mrbayes.con.tre'")
        found = self.opened()
        self.assertEqual((found["rooted"], found["rootStated"]), (False, True))
        self.assertEqual(found["treeName"], "con_50_majrule")

    def test_a_file_that_is_not_a_tree_says_where_the_error_is(self):
        self.start()
        self.open_text(text_of("broken.nwk"), "broken.nwk")
        self.wait_for("document.body.dataset.view === 'empty'")
        card = self.state("""({title: document.getElementById('empty-title').textContent, body: document.getElementById('empty-body').textContent,
            more: document.getElementById('empty-more').textContent, open: document.getElementById('empty-open').hidden,
            retry: document.getElementById('empty-retry').textContent.trim()})""")
        self.assertEqual(card["title"], "This is not a tree Tree View can read")
        self.assertEqual(card["body"], "broken.nwk could not be opened.")
        self.assertIn("A branch length is not a number", card["more"])
        self.assertIn("line 3, column 10", card["more"])
        self.assertFalse(card["open"])
        self.assertEqual(card["retry"], "Back to the tree")
        # The way back is the tree that was kept: the Viewer's.
        self.script("document.getElementById('empty-retry').click(); return 1")
        self.wait_for("document.body.dataset.view === 'tree'")
        self.assertEqual(self.state("TreeView.state.source"), "threshold")

    def test_other_files_that_are_no_tree_have_their_own_words(self):
        self.start()
        for text, words in ((">seq1\nMKTAYIAK\n", "looks like FASTA"), ("   \n", "The text is empty"), ("(A,B", "before every ( is closed"),
                            ("<html></html>", "neither PhyloXML nor NeXML"), ("{\"a\": 1}", "holds no tree"), ("{broken", "not well formed")):
            with self.subTest(text=text):
                self.open_text(text, "x.txt")
                self.wait_for("document.body.dataset.view === 'empty' && document.getElementById('empty-title').textContent.startsWith('This is not')")
                self.assertIn(words, self.state("document.getElementById('empty-more').textContent"))
                self.script("document.getElementById('empty-retry').click(); return 1")
                self.wait_for("document.body.dataset.view === 'tree'")
        self.assertEqual(self.console_errors(), [])


class OpeningTests(OpenCase):
    def test_the_chooser_has_an_entry_that_opens_a_file_and_lists_the_opened_tree_by_its_name(self):
        self.start()
        options = self.state("Array.from(document.getElementById('source-select').options).map(o => [o.value, o.textContent.trim()])")
        self.assertEqual([value for value, _ in options], ["threshold", "guide", "open"])
        self.assertEqual(options[2][1], "Open a tree file...")
        found = self.choose_file(text_of("primates.nwk"), "my primates.nwk")
        self.assertEqual(found["chooser"], 1)
        self.wait_uploaded()
        options = self.state("""Array.from(document.getElementById('source-select').options).map(o => [o.value, o.textContent.trim(), o.disabled])""")
        self.assertEqual([entry[0] for entry in options], ["threshold", "guide", "uploaded", "open"])
        self.assertEqual(options[2][1:], ["my primates.nwk", False])
        self.assertEqual(self.state("document.getElementById('source-select').value"), "uploaded")
        # The entry that opens a file is no tree: after it the list shows the tree that is shown.
        self.assertEqual(self.state("TreeView.state.source"), "uploaded")
        self.assertEqual(self.console_errors(), [])

    def test_a_file_dropped_on_the_page_opens(self):
        self.start()
        found = self.drop(text_of("nhx.nwk"), "dropped.nwk")
        self.assertEqual(found, {"dropping": True, "after": False})
        self.wait_uploaded()
        self.assertEqual(self.opened()["file"], "dropped.nwk")
        # Something dragged that is no file is left to the page.
        self.assertEqual(self.script("""const dt = new DataTransfer(); dt.setData('text/plain', 'x');
            const event = new DragEvent('dragover', {dataTransfer: dt, bubbles: true, cancelable: true}); document.dispatchEvent(event);
            return event.defaultPrevented"""), False)

    def test_tree_text_pasted_with_no_field_focused_opens(self):
        self.start()
        self.assertTrue(self.paste(text_of("primates.nwk")))
        self.wait_uploaded()
        found = self.opened()
        self.assertEqual((found["file"], found["matched"]), ("Pasted text", 5))

    def test_a_paste_into_a_field_is_the_fields(self):
        self.start()
        for target in ("document.getElementById('search-input')", "document.createElement('textarea')"):
            with self.subTest(target=target):
                self.assertFalse(self.paste(text_of("primates.nwk"), target))
        self.pump(0.4)
        self.assertEqual(self.state("TreeView.state.source"), "threshold")
        # Nothing to paste is nothing to open.
        self.assertFalse(self.paste("   "))
        self.assertEqual(self.state("TreeView.state.source"), "threshold")

    def test_the_viewer_keeps_the_tree_and_a_reload_shows_it(self):
        self.start()
        self.open_text(text_of("primates.nwk"), "kept.nwk")
        self.wait_uploaded()
        before = self.opened()
        self.open_page()
        self.assertEqual(self.state("TreeView.state.source"), "uploaded")
        after = self.opened()
        self.assertEqual((after["file"], after["labels"], after["matched"]), ("kept.nwk", before["labels"], before["matched"]))
        # The page does not show the card of the match again: that is for an opening.
        self.assertFalse(self.state("document.getElementById('match-report').classList.contains('shown')"))
        self.assertFalse(self.state("document.getElementById('report-btn').hidden"))
        self.assertEqual(self.state("document.getElementById('source-select').value"), "uploaded")

    def test_opening_another_tree_replaces_the_first_and_all_three_sources_can_be_switched_between(self):
        self.start(guide=True)
        self.assertEqual(self.state("TreeView.state.source"), "guide")
        self.open_text(text_of("primates.nwk"), "one.nwk")
        self.wait_uploaded()
        self.open_text(text_of("nested.json"), "two.json")
        self.wait_for("TreeView.state.source === 'uploaded' && TreeView.state.model.uploaded.file === 'two.json'")
        self.assertEqual(self.state("Array.from(document.getElementById('source-select').options).map(o => o.textContent.trim())"),
                         ["Network threshold tree", "MSA Guide Tree", "two.json", "Open a tree file..."])
        for source, leaves in (("threshold", len(NETWORK)), ("guide", len(NETWORK)), ("uploaded", 4), ("guide", len(NETWORK)), ("threshold", len(NETWORK))):
            with self.subTest(source):
                self.script(f"""const select = document.getElementById('source-select');
                    select.value = {json.dumps(source)}; select.dispatchEvent(new Event('change')); return 1""")
                self.wait_for(f"TreeView.state.source === {json.dumps(source)} && document.body.dataset.view === 'tree' && TreeView.state.model.source === {json.dumps(source)}")
                self.assertEqual(self.state("TreeView.state.model.leafCount"), leaves)
                self.assertEqual(self.viewer._tree_source, source)
        self.assertEqual(self.console_errors(), [])

    def test_the_viewer_refusing_a_tree_says_so_and_keeps_what_was_shown(self):
        self.start()
        self.open_text(text_of("primates.nwk"), "good.nwk")
        self.wait_uploaded()
        with mock.patch.object(tree_backend, "MAX_LEAVES", 3):
            self.open_text(text_of("primates.nwk"), "toolarge.nwk")
            self.wait_for("document.body.dataset.view === 'empty'")
        card = self.state("({title: document.getElementById('empty-title').textContent, body: document.getElementById('empty-body').textContent})")
        self.assertEqual(card["title"], "The Viewer did not take this tree")
        self.assertIn("toolarge.nwk was read, but: The tree has more than 3 leaves", card["body"])
        self.assertEqual(self.viewer._tree_upload.name, "good.nwk")
        self.script("document.getElementById('empty-retry').click(); return 1")
        self.wait_uploaded()
        self.assertEqual(self.opened()["file"], "good.nwk")

    def test_a_name_the_tree_has_twice_is_refused_with_the_name(self):
        self.start()
        self.open_text("((Alpha:1,Beta:1):1,(Alpha:1,Gamma:1):1);", "twice.nwk")
        self.wait_for("document.body.dataset.view === 'empty'")
        body = self.state("document.getElementById('empty-body').textContent")
        self.assertIn("Two leaves are called Alpha. A name must name one leaf", body)
        self.assertFalse(hasattr(self.viewer, "_tree_upload"))

    def test_what_is_hidden_is_not_drawn(self):
        # The button of the match's card is for a tree that was opened; a card with no path to copy has no box for one.
        self.start()
        shown = "(el) => getComputedStyle(document.getElementById(el)).display"
        self.assertEqual(self.state(f"({shown})('report-btn')"), "none")
        self.script("return await TreeView.loadTree('uploaded')")
        self.wait_for("document.body.dataset.view === 'empty'")
        self.assertEqual(self.state(f"({shown})('empty-path')"), "none")
        self.assertEqual(self.state(f"({shown})('report-btn')"), "none")
        self.open_text(text_of("primates.nwk"), "shown.nwk")
        self.wait_uploaded()
        self.assertNotEqual(self.state(f"({shown})('report-btn')"), "none")

    def test_the_tree_view_with_no_tree_opened_offers_to_open_one(self):
        self.start()
        self.script("""const select = document.getElementById('source-select'); return 1""")
        # Asked for before there is one, the page says so and offers the file chooser.
        self.script("return await TreeView.loadTree('uploaded')")
        self.wait_for("document.body.dataset.view === 'empty'")
        found = self.state("""({title: document.getElementById('empty-title').textContent, open: !document.getElementById('empty-open').hidden,
            label: document.getElementById('empty-open-label').textContent})""")
        self.assertEqual(found, {"title": "No tree has been opened", "open": True, "label": "Open a tree file"})


class MultipleTreesTests(OpenCase):
    def test_a_file_of_several_trees_asks_which(self):
        self.start()
        self.open_text(text_of("several.nex"), "several.nex")
        self.wait_for("document.body.dataset.view === 'empty' && !document.getElementById('empty-choice').hidden")
        card = self.state("""({title: document.getElementById('empty-title').textContent, body: document.getElementById('empty-body').textContent,
            choices: Array.from(document.getElementById('empty-choice').options).map(o => o.textContent), open: document.getElementById('empty-open-label').textContent})""")
        self.assertEqual(card["title"], "Choose a tree")
        self.assertEqual(card["body"], "several.nex holds 3 trees.")
        self.assertEqual(card["choices"], ["1. first", "2. second", "3. third"])
        self.assertEqual(card["open"], "Open this tree")
        # Nothing was sent to the Viewer yet.
        self.assertFalse(hasattr(self.viewer, "_tree_upload"))
        self.script("""const choice = document.getElementById('empty-choice'); choice.selectedIndex = 2;
            document.getElementById('empty-open').click(); return 1""")
        self.wait_uploaded()
        found = self.opened()
        self.assertEqual((found["treeName"], found["rooted"], found["rootStated"]), ("third", False, True))
        self.assertEqual(sorted(found["labels"]), ["A", "B", "C", "D"])

    def test_a_double_click_or_enter_opens_the_tree_of_the_list(self):
        self.start()
        self.open_text(text_of("several.nex"), "several.nex")
        self.wait_for("!document.getElementById('empty-choice').hidden")
        self.script("""const choice = document.getElementById('empty-choice'); choice.selectedIndex = 1;
            choice.dispatchEvent(new MouseEvent('dblclick', {bubbles: true})); return 1""")
        self.wait_uploaded()
        self.assertEqual(self.opened()["treeName"], "second")
        self.open_text(text_of("several.nex"), "several.nex")
        self.wait_for("!document.getElementById('empty-choice').hidden")
        self.script("""document.getElementById('empty-choice').dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true})); return 1""")
        self.wait_for("document.body.dataset.view === 'tree' && TreeView.state.model.uploaded.tree_name === 'first'")


class ReportTests(OpenCase):
    def test_the_card_of_the_match_follows_an_opening_and_counts_what_matched_how(self):
        headers = list(NETWORK)
        self.start(headers=headers)
        # Homo sapiens by the rules, the others exactly, one leaf the network does not have.
        text = "(('Homo sapiens':1,Pan_troglodytes:1):1,(Mus_musculus:1,Nobody_here:1):1,Bos_taurus:2);"
        self.open_text(text, "report.nwk")
        self.wait_uploaded()
        self.wait_for("document.getElementById('match-report').classList.contains('shown')")
        found = self.state("""({text: document.getElementById('match-report').textContent,
            box: document.querySelector('#match-report textarea').value, copy: !!document.getElementById('report-copy'),
            button: !document.getElementById('report-btn').hidden && !document.getElementById('report-btn').disabled})""")
        for words in ("4 of the 5 leaves of report.nwk are nodes of this network.", "Matched by the exact full name: 3",
                      "Matched by the name after the Viewer's header rules: 1", "Leaves that are not in the network: 1",
                      f"Nodes of the network that are not in the tree: {len(headers) - 4}"):
            self.assertIn(words, found["text"])
        self.assertEqual(found["box"], "Nobody_here")
        self.assertTrue(found["copy"] and found["button"])
        # The card closes, and the Match button brings it back.
        self.script("document.querySelector('#match-report .head button').click(); return 1")
        self.assertFalse(self.state("document.getElementById('match-report').classList.contains('shown')"))
        self.script("document.getElementById('report-btn').click(); return 1")
        self.assertTrue(self.state("document.getElementById('match-report').classList.contains('shown')"))
        self.assertEqual(self.console_errors(), [])
        self.shot("report-card.png")

    def test_the_toolbar_is_one_row_at_1300_pixels_in_every_state(self):
        self.start(guide=True)
        rows = """(() => { const bar = document.querySelector('.toolbar');
            const tops = new Set(Array.from(bar.children).filter(e => e.offsetParent !== null && e.getBoundingClientRect().width > 2).map(e => { const r = e.getBoundingClientRect(); return Math.round((r.top + r.bottom) / 2 / 20); }));
            const header = document.querySelector('header').getBoundingClientRect().height;
            return {tops: Array.from(tops), width: window.innerWidth, header}; })()"""
        self.assertEqual(self.state("window.innerWidth"), 1300)
        for label, action in (
            ("guide", lambda: None),
            ("threshold", lambda: self.script("await TreeView.loadTree('threshold'); return 1")),
            ("uploaded", lambda: (self.open_text(text_of("primates.nwk"), "p.nwk"), self.wait_uploaded())),
            ("empty", lambda: (self.open_text(">x\nAC", "x.fa"), self.wait_for("document.body.dataset.view === 'empty'"))),
        ):
            with self.subTest(label):
                action()
                self.pump(0.3)
                found = self.state(rows)
                self.assertEqual(len(found["tops"]), 1, found)
        # With a tree open the Match button is in the header, beside the Tree list.
        self.assertEqual(self.state("document.getElementById('report-btn').closest('header') !== null"), True)

    def test_the_list_of_leaves_that_are_not_in_the_network_is_copyable_whole(self):
        self.start()
        names = [f"Missing_{k}" for k in range(5030)]
        newick = "(" + ",".join(f"{name}:1" for name in names) + ",Alpha:1,Beta:1);"
        self.open_text(newick, "many.nwk")
        self.wait_uploaded()
        self.wait_for("document.getElementById('match-report').classList.contains('shown')")
        found = self.script("""
            let copied = null;
            Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {writeText: (text) => { copied = text; return Promise.resolve(); }}});
            const box = document.querySelector('#match-report textarea').value.split('\\n');
            document.getElementById('report-copy').click();
            await new Promise((resolve) => setTimeout(resolve, 100));
            return {lines: box.length, last: box[box.length - 1], copied: copied.split('\\n').length, button: document.getElementById('report-copy').textContent.trim()};""")
        self.assertEqual(found["copied"], len(names))
        self.assertEqual(found["button"], "Copied")
        # The box lists 5,000 and says how many more; the copy has every one.
        self.assertEqual((found["lines"], found["last"]), (5001, "... and 30 more"))

    def test_a_tree_that_matches_everything_has_no_list_to_copy(self):
        self.start()
        self.open_text(text_of("primates.nwk"), "all.nwk")
        self.wait_uploaded()
        self.wait_for("document.getElementById('match-report').classList.contains('shown')")
        self.assertEqual(self.state("document.querySelector('#match-report textarea')"), None)
        self.assertIn("5 of the 5 leaves", self.state("document.getElementById('match-report').textContent"))

    def test_two_names_that_reach_one_node_are_left_unmatched_and_counted(self):
        self.start()
        self.open_text("((Alpha:1,'Alpha ':1):1,(Beta:1,Gamma:1):1);", "pair.nwk")
        self.wait_uploaded()
        found = self.opened()
        self.assertEqual(found["matched"], 3)
        self.assertEqual(found["unmatched"], 1)
        self.assertIn("ambiguous_names", found["warnings"])
        self.assertIn("left unmatched: 1", self.state("document.getElementById('banner').textContent"))

    def test_with_no_match_at_all_the_tree_is_not_shown_and_names_stand_beside_headers(self):
        self.start()
        renamed = "(" + ",".join(f"{name}_0:1" for name in PRIMATES[:3]) + ");"  # a program's own names for the sequences
        self.open_text(renamed, "renamed.treefile")
        self.wait_for("document.body.dataset.view === 'empty' && document.getElementById('empty-title').textContent.startsWith('No leaf')")
        found = self.state("""({body: document.getElementById('empty-body').textContent, report: document.getElementById('empty-report').textContent,
            lists: Array.from(document.querySelectorAll('#empty-report ul')).map(ul => Array.from(ul.children).map(li => li.textContent)),
            canvasModel: TreeView.state.model, open: document.getElementById('empty-open-label').textContent,
            select: document.getElementById('source-select').value})""")
        self.assertIsNone(found["canvasModel"])
        self.assertIn("None of the 3 leaves of renamed.treefile has the full name of a node of this network", found["body"])
        self.assertEqual(found["lists"][0], ["Homo_sapiens_0", "Pan_troglodytes_0", "Mus_musculus_0"])
        self.assertEqual(found["lists"][1], NETWORK[:5])
        self.assertIn("Names in the tree", found["report"])
        self.assertIn("Headers in the network", found["report"])
        self.assertEqual(found["open"], "Open another tree file")
        # The chooser still lists it by its name, and a reload shows the same card.
        self.assertEqual(found["select"], "uploaded")
        self.open_page("empty")
        self.assertTrue(self.state("document.getElementById('empty-title').textContent").startswith("No leaf of this tree"))
        self.shot("no-match-card.png")

    def test_a_tree_the_page_has_no_words_for_a_warning_of_is_still_drawn(self):
        self.start()
        self.open_text("((Alpha:-1,Beta:1):1,(Gamma:1,Delta:1):1)", "odd.nwk")
        self.wait_uploaded()
        banner = self.state("document.getElementById('banner').textContent")
        self.assertIn("Branches with a negative length, drawn with length 0: 1.", banner)
        self.assertIn("does not end with a ;", banner)
        # The negative length was sent as 0.
        self.assertEqual(float(self.viewer._tree_upload.length.min()), 0.0)
        # The Details say what the reader noted, too.
        self.script("document.getElementById('details-btn').click(); return 1")
        details = self.state("document.getElementById('details').textContent")
        self.assertIn("What the reader noted", details)
        self.assertIn("Branches with a negative length, drawn with length 0: 1.", details)


class DrawingTests(OpenCase):
    def test_leaves_the_network_lacks_are_drawn_faded_as_not_a_node_of_the_network(self):
        self.start()
        self.open_text("(((Alpha:1,Beta:1):1,(Gamma:1,Nowhere_1:1):1):1,((Delta:1,Nowhere_2:1):1,Nowhere_3:2):1);", "faded.nwk")
        self.wait_uploaded()
        found = self.script("""
            const s = TreeView.state, m = s.model;
            TreeView.zoomToNode(m.root);
            s.view.sx *= 50; s.view.ox = 900 - s.layout.width * s.view.sx; s.view.sy = 30; s.view.oy = 0.5; TreeView.drawNow();
            const ctx = document.createElement('canvas').getContext('2d');
            const rgb = css => { ctx.fillStyle = css; ctx.fillRect(0, 0, 1, 1); return Array.from(ctx.getImageData(0, 0, 1, 1).data.slice(0, 3)); };
            const dots = {};
            for (let k = 0; k < m.leafCount; k++) {
                const box = TreeView.nodePixels(s, m.leafNode[k]);
                dots[m.labels[k]] = fire.pixel(box.x, box.y).slice(0, 3);
            }
            return {dots, shown: rgb(s.palette.branch), faded: rgb(s.palette.faded),
                    tips: Array.from(m.labels, (name, k) => [name, TreeView.describeNode(m.leafNode[k]).lines[0]])};""")
        for name, dot in found["dots"].items():
            expected = found["faded"] if name.startswith("Nowhere") else found["shown"]
            with self.subTest(name):
                self.assertEqual(dot, expected)
        lines = dict(found["tips"])
        self.assertEqual(lines["Alpha"], "A node of this network")
        self.assertEqual(lines["Nowhere_2"], "Not a node of this network")

    def test_colours_selection_and_hiding_work_on_an_opened_tree(self):
        self.start(colors=True, hidden=[3], select=[0, 1])
        self.wait_for("TreeView.state.colors !== null")
        self.open_text(text_of("primates.nwk"), "colours.nwk")
        self.wait_uploaded()
        # Node 1 is Pan_troglodytes (red); the selection is nodes 0 and 1.
        self.wait_for("TreeView.state.colors !== null && TreeView.state.selection.count === 2")
        found = self.state("""(() => { const s = TreeView.state, m = s.model;
            const leafOf = (node) => m.viewerNode.indexOf(node);
            return {legend: s.colors.legend.map(e => e.css), red: s.colors.leafClass[leafOf(1)], plain: s.colors.leafClass[leafOf(0)],
                    selected: Array.from(s.selection.leaves, (k) => m.labels[k]).sort(), removed: m.labels.filter((_, k) => s.removed && s.removed[k]),
                    summary: document.getElementById('summary').textContent}; })()""")
        self.assertEqual(found["selected"], ["Homo_sapiens", "Pan_troglodytes"])
        self.assertNotEqual(found["red"], found["plain"])
        self.assertEqual(found["removed"], ["Rattus_norvegicus"])
        self.assertIn("2 selected in the network", found["summary"])
        self.assertIn("1 hidden in the network, left out", found["summary"])
        # A change in the network reaches the opened tree: select another node, and hide another.
        self.viewer.select([2])
        self.wait_for("TreeView.state.selection.count === 1")
        self.assertEqual(self.state("Array.from(TreeView.state.selection.leaves, (k) => TreeView.state.model.labels[k])"), ["Mus_musculus"])
        # Selecting a clade of the opened tree selects the network's nodes of its leaves, as it does for any tree.
        self.script("""const m = TreeView.state.model; let top = -1;
            for (let i = 0; i < m.nodeCount - 1; i++) if (m.childCount[i] > 0 && m.leavesCount === undefined) { top = i; break; }
            await TreeView.selectClade(top, 'replace'); return top""")
        self.wait_for("TreeView.state.selection.count === 2")
        self.assertEqual(sorted(self.viewer.selected_indices), [0, 1])
        self.assertEqual(self.console_errors(), [])

    def test_a_tree_with_support_draws_dots_and_words_the_legend_for_the_files_support(self):
        self.start()
        self.open_text(text_of("primates.nwk"), "support.nwk")
        self.wait_uploaded()
        found = self.state("""({hidden: document.getElementById('support-legend').hidden, title: document.getElementById('support-title').textContent,
            note: document.getElementById('support-note').textContent, tip: document.getElementById('support-btn').title,
            line: (() => { const m = TreeView.state.model; let node = -1;
                for (let i = 0; i < m.nodeCount - 1; i++) if (m.childCount[i] > 0 && m.support[i] >= 0) { node = i; break; }
                return TreeView.describeNode(node).lines.filter(l => l.startsWith('Support') || l.startsWith('Stable')); })()})""")
        self.assertFalse(found["hidden"])
        self.assertEqual(found["title"], "Support")
        self.assertIn("the file gives each split", found["note"])
        self.assertIn("file gives", found["tip"])
        self.assertEqual(len(found["line"]), 1)
        self.assertTrue(found["line"][0].startswith("Support: "))
        # Back on the guide tree's wording when another tree is shown.
        self.script("""const select = document.getElementById('source-select'); select.value = 'threshold'; select.dispatchEvent(new Event('change')); return 1""")
        self.wait_for("TreeView.state.model.source === 'threshold'")
        self.assertTrue(self.state("document.getElementById('support-legend').hidden"))
        self.script("""const select = document.getElementById('source-select'); select.value = 'uploaded'; select.dispatchEvent(new Event('change')); return 1""")
        self.wait_for("TreeView.state.model.source === 'uploaded'")
        self.script("""document.getElementById('support-btn').click(); return TreeView.drawNow().supports""")
        self.assertGreater(self.state("TreeView.drawNow().supports"), 0)

    def test_the_details_say_what_the_file_is(self):
        self.start()
        self.open_text(text_of("iqtree.treefile"), "iqtree.treefile")
        self.wait_uploaded()
        self.script("document.getElementById('details-btn').click(); return 1")
        text = self.state("document.getElementById('details').textContent")
        for words in ("iqtree.treefile", "Newick", "Leaves5", "Rooting", "Unrooted (its root has three or more branches)",
                      "Branch lengthsYes", "Leaves in this network5", "Matched by the exact name5",
                      "Matched by the name after the Viewer's header rules0", "Network nodes not in the tree", "kept in the Viewer until it is closed"):
            self.assertIn(words, text)
        self.assertNotIn("Alignment", text)
        # An unrooted tree is drawn from the root the file writes, and the note says so.
        self.script("""const select = document.getElementById('layout-select'); select.value = 'rectangular';
            select.dispatchEvent(new Event('change')); return 1""")
        self.assertEqual(self.state("document.querySelector('.rooting-note span').textContent"),
                         "Drawn from the root the file writes; the tree itself is unrooted.")
        self.assertIn("the file writes", self.state("document.getElementById('details-rooting').textContent"))

    def layout(self, kind):
        self.script(f"TreeView.setLayout({json.dumps(kind)}, TreeView.state.mode); TreeView.drawNow(); return 1")

    def test_the_support_of_the_file_is_not_called_stability_in_the_details(self):
        self.start()
        self.open_text(text_of("primates.nwk"), "s.nwk")
        self.wait_uploaded()
        self.script("document.getElementById('details-btn').click(); return 1")
        text = self.state("document.getElementById('details').textContent")
        self.assertIn("not how stable the split is under noise", text)

    def test_the_layouts_draw_an_opened_tree(self):
        self.start()
        self.open_text(text_of("beast.tre"), "beast.tre")
        self.wait_uploaded()
        for kind in ("rectangular", "circular", "unrooted"):
            with self.subTest(kind):
                self.layout(kind)
                self.assertGreater(self.state("TreeView.drawNow().visited"), 0)
        self.assertEqual(self.console_errors(), [])
        self.shot("opened-tree.png")


class SpeedTests(OpenCase):
    def test_a_tree_of_44000_leaves_goes_from_text_to_first_frame_in_about_a_second(self):
        count = 44000
        rng = np.random.default_rng(42)
        headers = [f"sp|Q{k:06d}|PROTEIN_{k}" for k in range(count)]
        # A random binary tree: join random pairs until one is left, written as Newick (every branch has a length).
        pool = [f"{header}:{rng.random():.4f}" for header in headers]
        while len(pool) > 1:
            rng.shuffle(pool)
            joined = [f"({pool[k]},{pool[k + 1]}):{rng.random():.4f}" for k in range(0, len(pool) - 1, 2)]
            if len(pool) % 2:
                joined.append(pool[-1])
            pool = joined
        newick = pool[0].rsplit(":", 1)[0] + ";"
        self.serve_headers(headers[:30], network=headers, tree=None)
        give_edges(self.viewer, [(n, n + 1) for n in range(count - 1)], np.arange(count - 1, 0, -1), threshold=3.5)
        self.open_page()
        self.js(f"window.__newick = {json.dumps(newick)}; 1")
        self.script("""window.__t0 = performance.now(); await TreeView.openText(window.__newick, {name: 'big.nwk', size: window.__newick.length, pasted: false}); return 1""")
        self.wait_for("TreeView.state.source === 'uploaded' && document.body.dataset.view === 'tree'")
        elapsed = self.state("[performance.now() - window.__t0, TreeView.lastOpening]")
        total, parts = elapsed
        # The frame is drawn when the page says the tree is shown: applyPayload draws it.
        self.assertEqual(self.state("TreeView.state.model.leafCount"), count)
        self.assertEqual(self.state("TreeView.state.model.match.matched"), count)
        print(f"\n44,000-leaf upload: read {parts['read']:.0f} ms, Viewer {parts['answered']:.0f} ms, drawn {parts['drawn']:.0f} ms (to the page's quiet: {total:.0f} ms)")
        self.assertLess(parts["read"] + parts["answered"] + parts["drawn"], 3000)


class BannerCase(OpenCase):
    """The banner of warnings: one line that rolls, and the card of the leaves and nodes it counts."""

    TREE_LEAVES = 30
    LINE = "document.getElementById('banner-line')"

    def guide(self, *, tree_leaves=30, network_extra=5, shared=20, **options):
        """A page over a guide tree of ``tree_leaves`` leaves, of which ``shared`` are nodes of the network, which also
        has ``network_extra`` nodes the tree has no leaf for: two warnings that name leaves and nodes."""
        headers = headers_of(tree_leaves)
        network = headers[:shared] + headers_of(network_extra, prefix="tr|Y")
        self.serve_headers(headers, network=network, **options)
        give_edges(self.viewer, [(n, n + 1) for n in range(len(network) - 1)], np.arange(len(network) - 1, 0, -1), threshold=3.5)
        self.enable_selection()
        self.open_page()
        self.headers, self.network = headers, network
        return headers, network

    def resize(self, width):
        self.view.resize(width, 800)
        self.addCleanup(self.view.resize, 1300, 800)
        self.pump(0.3)

    def banner(self):
        return self.script(f"""
            const line = {self.LINE}, track = document.getElementById('warning-list');
            const items = [...track.children];
            return {{shown: document.getElementById('banner').classList.contains('shown'), rolls: line.classList.contains('rolls'),
                texts: items.map(li => li.textContent), tops: items.map(li => li.getBoundingClientRect().top),
                height: document.getElementById('banner').getBoundingClientRect().height, title: line.title,
                room: line.clientWidth, text: items.length ? items[items.length - 1].offsetLeft + items[items.length - 1].offsetWidth - items[0].offsetLeft : 0,
                animations: track.getAnimations().length, transform: getComputedStyle(track).transform,
                button: !document.getElementById('mismatch-btn').hidden, tabindex: line.tabIndex, label: line.getAttribute('aria-label')}};""")

    def set_reduced_motion(self, reduce):
        """Say what the page's media query for reduced motion answers (the browser's own depends on the desktop's settings)."""
        self.script(f"""const real = window.__matchMedia || window.matchMedia.bind(window); window.__matchMedia = real;
            window.matchMedia = (query) => query.includes('prefers-reduced-motion') ? {{matches: {str(bool(reduce)).lower()}, media: query}} : real(query);
            return 1""")

    def overflow_by(self, extra):
        """Make the window so wide that the sentences are longer than the line's room by ``extra`` pixels."""
        self.set_reduced_motion(False)
        found = self.banner()
        width = self.state("window.innerWidth") - (found["room"] - (found["text"] - extra))
        self.resize(int(width))
        self.wait_for(f"document.getElementById('banner-line').classList.contains('rolls')")
        return self.banner()

    def hover(self, on=True, target=None):
        target = target or self.LINE
        kind = "pointerenter" if on else "pointerleave"
        self.script(f"{target}.dispatchEvent(new PointerEvent('{kind}', {{pointerId: 3}})); return 1")

    def translate_x(self):
        return self.state("new DOMMatrix(getComputedStyle(document.getElementById('warning-list')).transform).m41")


class BannerLineTests(BannerCase):
    """All the sentences are one line, which rolls under the pointer when it is longer than its room."""

    def unrooted_upload(self):
        """An opened tree with warnings and an unrooted tree drawn from a root: the sentences and the note."""
        self.start()
        self.open_text("((Alpha:1,Beta:1):1,(Gamma:1,Nope:1):1,Nope2:1)", "odd.nwk")
        self.wait_uploaded()
        self.script("""const select = document.getElementById('layout-select');
            select.value = 'rectangular'; select.dispatchEvent(new Event('change')); return 1""")

    def test_every_sentence_and_the_note_stand_in_one_line_as_tall_as_one_sentence(self):
        self.unrooted_upload()
        found = self.banner()
        self.assertEqual(len(found["texts"]), 4)
        self.assertEqual(found["texts"][-1], "Drawn from the root the file writes; the tree itself is unrooted.")
        self.assertEqual(len(set(found["tops"])), 1, found["tops"])
        many = found["height"]
        self.script("""const select = document.getElementById('layout-select');
            select.value = 'unrooted'; select.dispatchEvent(new Event('change')); return 1""")
        fewer = self.banner()
        self.assertEqual(len(fewer["texts"]), 3)
        self.assertEqual(fewer["height"], many)
        self.assertLess(many, 40)
        self.assertEqual(self.console_errors(), [])

    def test_the_sentences_have_a_gap_and_the_whole_text_is_the_lines_title(self):
        self.unrooted_upload()
        found = self.banner()
        self.assertEqual(found["title"], "  ·  ".join(found["texts"]))
        gaps = self.state("""(() => { const items = [...document.getElementById('warning-list').children];
            return items.slice(1).map((li, k) => li.offsetLeft - (items[k].offsetLeft + items[k].offsetWidth)); })()""")
        self.assertTrue(all(gap >= 24 for gap in gaps), gaps)

    def test_when_everything_fits_nothing_moves(self):
        self.guide()
        found = self.banner()
        self.assertEqual(len(found["texts"]), 2)
        self.assertFalse(found["rolls"])
        self.hover()
        self.pump(0.3)
        found = self.banner()
        self.assertEqual((found["animations"], found["transform"]), (0, "none"))
        self.hover(False)

    def test_a_line_longer_than_its_room_is_clipped_and_rolls_while_the_pointer_is_over_it(self):
        self.guide()
        found = self.overflow_by(100)
        self.assertEqual((found["tabindex"], found["label"]), (0, "Warnings about the tree"))
        self.assertEqual(found["animations"], 0)
        self.hover()
        self.wait_for("document.getElementById('warning-list').getAnimations().length === 1")
        distance = found["text"] + 28 - found["room"]
        timing = self.state("""(() => { const a = document.getElementById('warning-list').getAnimations()[0];
            const t = a.effect.getComputedTiming(); return {duration: t.duration, iterations: t.iterations, state: a.playState}; })()""")
        slide = distance / 50 * 1000
        self.assertAlmostEqual(timing["duration"], 2 * slide + 1500, delta=1)
        self.assertEqual(timing["state"], "running")
        # A steady speed: about 50 pixels a second.
        self.script("document.getElementById('warning-list').getAnimations()[0].currentTime = 200; return 1")
        first = self.translate_x()
        self.pump(0.5)
        moved = first - self.translate_x()
        self.assertGreater(moved, 10)
        self.assertLess(moved, 45)
        # It reaches the end, holds there for the hold, comes back, and starts again.
        for at, wanted in ((slide, -distance), (slide + 1000, -distance), (2 * slide + 1500 - 1, 0)):
            self.script(f"""const a = document.getElementById('warning-list').getAnimations()[0]; a.pause(); a.currentTime = {at}; return 1""")
            self.assertAlmostEqual(self.translate_x(), wanted, delta=1.5)
        # The last sentence is wholly in view at the end, clear of the fade.
        self.script(f"""const a = document.getElementById('warning-list').getAnimations()[0]; a.currentTime = {slide + 100}; return 1""")
        edge = self.state("""(() => { const line = document.getElementById('banner-line').getBoundingClientRect();
            const last = document.getElementById('warning-list').lastElementChild.getBoundingClientRect();
            return line.right - last.right; })()""")
        self.assertGreaterEqual(edge, 27)
        # The pointer leaves: the roll ends and the line is at its start.
        self.hover(False)
        found = self.banner()
        self.assertEqual((found["animations"], found["transform"]), (0, "none"))
        self.assertEqual(self.console_errors(), [])

    def test_the_keyboard_rolls_the_line_too(self):
        self.guide()
        self.overflow_by(100)
        self.script(f"{self.LINE}.dispatchEvent(new FocusEvent('focusin', {{bubbles: true}})); return 1")
        self.assertEqual(self.banner()["animations"], 1)
        self.script(f"{self.LINE}.dispatchEvent(new FocusEvent('focusout', {{bubbles: true}})); return 1")
        self.assertEqual(self.banner()["animations"], 0)

    def test_reduced_motion_clips_the_line_and_does_not_roll_it(self):
        self.guide()
        found = self.overflow_by(100)
        self.set_reduced_motion(True)
        self.hover()
        self.pump(0.3)
        after = self.banner()
        self.assertTrue(after["rolls"])
        self.assertEqual((after["animations"], after["transform"]), (0, "none"))
        self.assertEqual(after["title"], found["title"])
        self.assertIn("Nodes of this network that are not leaves of the tree: 5.", after["title"])

    def test_the_line_is_measured_again_when_the_window_or_the_words_change(self):
        self.guide()
        self.overflow_by(100)
        self.resize(1300)
        self.wait_for(f"!{self.LINE}.classList.contains('rolls')")
        # A language whose words are longer: the text of a sentence changes under the page's feet.
        self.script("""document.querySelector('#warning-list li span').textContent += ' ' + 'long words '.repeat(60); return 1""")
        self.wait_for(f"{self.LINE}.classList.contains('rolls')")
        self.script("document.querySelector('#warning-list li span').textContent = 'Short.'; return 1")
        self.wait_for(f"!{self.LINE}.classList.contains('rolls')")

    def test_the_x_waits_the_roll_is_clickable_and_hides_only_the_note(self):
        self.unrooted_upload()
        self.overflow_by(100)
        self.hover()
        self.wait_for("document.getElementById('warning-list').getAnimations().length === 1")
        self.hover(True, "document.querySelector('.note-close')")
        self.assertEqual(self.state("document.getElementById('warning-list').getAnimations()[0].playState"), "paused")
        shape = self.state("""(() => { const x = document.querySelector('.note-close');
            return {after: x.previousElementSibling.textContent, last: x.parentElement.lastElementChild === x, parent: x.parentElement.className}; })()""")
        self.assertEqual(shape["parent"], "rooting-note")
        self.assertTrue(shape["last"])
        self.assertTrue(shape["after"].startswith("Drawn from the root"))
        self.hover(False, "document.querySelector('.note-close')")
        self.assertEqual(self.state("document.getElementById('warning-list').getAnimations()[0].playState"), "running")
        self.script("document.querySelector('.note-close').click(); return 1")
        found = self.banner()
        self.assertEqual(len(found["texts"]), 3)
        self.assertNotIn("Drawn from", " ".join(found["texts"]))
        self.assertTrue(found["shown"])

    def test_a_key_that_reaches_the_x_shows_it(self):
        self.unrooted_upload()
        found = self.overflow_by(100)
        self.script("document.querySelector('.note-close').dispatchEvent(new FocusEvent('focus')); return 1")
        self.assertAlmostEqual(self.translate_x(), -(found["text"] + 28 - found["room"]), delta=1.5)
        right = self.state("""(() => document.getElementById('banner-line').getBoundingClientRect().right
            - document.querySelector('.note-close').getBoundingClientRect().right)()""")
        self.assertGreaterEqual(right, 27)


class ShowDetailsTests(BannerCase):
    """The button at the right end of the banner and the card of the leaves and nodes the warnings count."""

    def card(self):
        return self.script("""
            const card = document.getElementById('mismatch-card');
            return {shown: card.classList.contains('shown'), expanded: document.getElementById('mismatch-btn').getAttribute('aria-expanded'),
                headings: [...card.querySelectorAll('.group-title')].map(e => e.textContent),
                boxes: [...card.querySelectorAll('textarea')].map(e => e.value.split('\\n')),
                copies: [...card.querySelectorAll('.mismatch-copy')].map(e => e.textContent.trim()),
                title: card.querySelector('h2') && card.querySelector('h2').textContent};""")

    def wait_card(self, lists):
        self.wait_for(f"document.querySelectorAll('#mismatch-card textarea').length === {lists}")
        return self.card()

    def press(self):
        self.script("document.getElementById('mismatch-btn').click(); return 1")

    def test_the_button_is_at_the_right_end_and_is_shown_only_for_the_four_warnings(self):
        self.guide(shared=30, network_extra=0)
        self.assertFalse(self.banner()["shown"])
        self.assertFalse(self.banner()["button"])
        # A tree made for another alignment: a warning about the files, not about leaves or nodes.
        self.guide(notes=[("ALIGNMENT", "Other_[model]_alignment.fasta")], shared=30, network_extra=0)
        found = self.banner()
        self.assertEqual(len(found["texts"]), 1)
        self.assertFalse(found["button"])
        # Leaves the alignment lacks.
        self.guide(shared=30, network_extra=0)
        self.viewer.alignment.aligned_node_mask[:3] = False
        self.open_page()
        found = self.banner()
        self.assertEqual(found["texts"], ["Leaves whose sequence is not in the loaded alignment: 3."])
        self.assertTrue(found["button"])
        # Leaves and nodes: both.
        self.guide()
        self.assertTrue(self.banner()["button"])
        place = self.state("""(() => { const bar = document.getElementById('banner').getBoundingClientRect(),
            button = document.getElementById('mismatch-btn').getBoundingClientRect(), line = document.getElementById('banner-line').getBoundingClientRect();
            return {gap: bar.right - button.right, after: button.left >= line.right, label: document.getElementById('mismatch-btn').textContent.trim(),
                title: document.getElementById('mismatch-btn').title, expanded: document.getElementById('mismatch-btn').getAttribute('aria-expanded')}; })()""")
        self.assertLessEqual(place["gap"], 17)
        self.assertTrue(place["after"])
        self.assertEqual((place["label"], place["expanded"]), ("Show Details", "false"))
        self.assertTrue(place["title"])

    def test_the_button_does_not_roll_and_the_banner_is_as_tall_with_it_as_without(self):
        self.guide()
        with_button = self.banner()["height"]
        self.overflow_by(100)
        self.hover()
        before = self.state("document.getElementById('mismatch-btn').getBoundingClientRect().left")
        self.pump(0.4)
        self.assertEqual(self.state("document.getElementById('mismatch-btn').getBoundingClientRect().left"), before)
        self.guide(shared=30, network_extra=0, notes=[("ALIGNMENT", "Other_[model]_alignment.fasta")])
        self.assertEqual(self.banner()["height"], with_button)

    def test_the_card_lists_each_warnings_names_under_a_heading_with_the_count(self):
        headers, network = self.guide()
        self.viewer.alignment.aligned_node_mask[[0, 3, 22]] = False
        self.open_page()
        self.press()
        found = self.wait_card(3)
        self.assertEqual((found["shown"], found["expanded"]), (True, "true"))
        self.assertEqual(found["headings"], [
            "Leaves that are not in the network: 10",
            "Nodes of the network that are not in the tree: 5",
            "Leaves whose sequence is not in the loaded alignment: 2",
        ])
        self.assertEqual(sorted(found["boxes"][0]), sorted(headers[20:]))
        self.assertEqual(found["boxes"][1], network[20:])
        self.assertEqual(sorted(found["boxes"][2]), sorted([network[0], network[3]]))
        self.assertEqual(found["copies"], ["Copy the list"] * 3)
        self.assertEqual(self.console_errors(), [])

    def test_the_button_card_is_closed_by_the_button_escape_and_the_other_cards(self):
        self.guide()
        self.press()
        self.wait_card(2)
        self.press()
        found = self.card()
        self.assertEqual((found["shown"], found["expanded"]), (False, "false"))
        self.press()
        self.wait_card(2)
        self.script("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true})); return 1")
        self.assertFalse(self.card()["shown"])
        self.assertEqual(self.card()["expanded"], "false")
        # Details closes it, and it closes Details.
        self.press()
        self.wait_card(2)
        self.script("document.getElementById('details-btn').click(); return 1")
        self.assertEqual(self.card()["expanded"], "false")
        self.assertFalse(self.card()["shown"])
        self.assertTrue(self.state("document.getElementById('details').classList.contains('shown')"))
        self.press()
        self.assertTrue(self.card()["shown"])
        self.assertFalse(self.state("document.getElementById('details').classList.contains('shown')"))
        # Its own x.
        self.wait_card(2)
        self.script("document.querySelector('#mismatch-card .head .btn').click(); return 1")
        self.assertFalse(self.card()["shown"])

    def test_the_card_closes_the_card_of_the_match_and_the_match_closes_it(self):
        self.start()
        self.open_text("((Alpha:1,Beta:1):1,(Gamma:1,Nope:1):1,Nope2:1);", "odd.nwk")
        self.wait_uploaded()
        self.wait_for("document.getElementById('match-report').classList.contains('shown')")
        self.press()
        self.wait_for("document.getElementById('mismatch-card').classList.contains('shown')")
        self.assertFalse(self.state("document.getElementById('match-report').classList.contains('shown')"))
        self.script("document.getElementById('report-btn').click(); return 1")
        self.assertTrue(self.state("document.getElementById('match-report').classList.contains('shown')"))
        self.assertFalse(self.card()["shown"])
        self.assertEqual(self.card()["expanded"], "false")

    def test_an_opened_tree_lists_its_leaves_and_the_nodes_two_names_reach(self):
        self.start(headers=["Alpha", "Beta", "Gamma", "Delta", "Epsilon"])
        self.open_text("((Alpha:1,Beta:1):1,(Gamma:1,'Gamma ':1):1,Nope:1)", "odd.nwk")
        self.wait_uploaded()
        self.press()
        found = self.wait_card(3)
        self.assertEqual(found["headings"], [
            "Leaves that are not in the network: 2",
            "Nodes of the network that are not in the tree: 2",
            "Leaves whose name reaches a node that another leaf has too, left unmatched: 1",
        ])
        self.assertEqual(found["boxes"], [["Gamma ", "Nope"], ["Delta", "Epsilon"], ["Gamma "]])

    def test_copy_has_every_name_though_the_box_lists_five_thousand(self):
        extra = [f"extra_{k}" for k in range(5100)]
        self.start(headers=NETWORK + extra)
        self.open_text(text_of("primates.nwk"), "p.nwk")
        self.wait_uploaded()
        self.press()
        self.wait_for("document.querySelectorAll('#mismatch-card textarea').length >= 1")
        found = self.script("""
            let copied = null;
            Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {writeText: (text) => { copied = text; return Promise.resolve(); }}});
            const box = document.querySelector('#mismatch-card textarea').value.split('\\n');
            const heading = document.querySelector('#mismatch-card .group-title').textContent;
            document.querySelector('#mismatch-card .mismatch-copy').click();
            await new Promise((resolve) => setTimeout(resolve, 100));
            return {heading, lines: box.length, last: box[box.length - 1], copied: copied.split('\\n').length,
                button: document.querySelector('#mismatch-card .mismatch-copy').textContent.trim()};""")
        total = len(NETWORK) + len(extra) - 5
        self.assertEqual(found["heading"], f"Nodes of the network that are not in the tree: {total:,}")
        self.assertEqual(found["copied"], total)
        self.assertEqual(found["button"], "Copied")
        self.assertEqual((found["lines"], found["last"]), (5001, f"... and {total - 5000} more"))

    def test_the_card_follows_a_change_of_source_and_goes_when_there_is_nothing_to_list(self):
        headers, network = self.guide()
        self.press()
        guide = self.wait_card(2)
        self.assertEqual(guide["headings"][0], "Leaves that are not in the network: 10")
        # Another tree is opened while the card is open: the card shows that tree's names.
        self.open_text(f"(('{network[0]}':1,'{network[1]}':1):1,Nope:1)", "odd.nwk")
        self.wait_uploaded()
        self.assertFalse(self.card()["shown"])
        self.press()
        uploaded = self.wait_card(2)
        self.assertEqual(uploaded["headings"][0], "Leaves that are not in the network: 1")
        self.assertEqual(uploaded["boxes"][0], ["Nope"])
        # The chooser goes back to the guide tree with the card open: it asks again and shows the guide tree's names.
        self.script("""const select = document.getElementById('source-select');
            select.value = 'guide'; select.dispatchEvent(new Event('change')); return 1""")
        self.wait_for("TreeView.state.source === 'guide'")
        again = self.wait_card(2)
        self.assertTrue(again["shown"])
        self.assertEqual(again["headings"], guide["headings"])
        self.assertEqual(sorted(again["boxes"][0]), sorted(headers[20:]))
        # The threshold tree has no warning: the button and the card go.
        self.script("""const select = document.getElementById('source-select');
            select.value = 'threshold'; select.dispatchEvent(new Event('change')); return 1""")
        self.wait_for("TreeView.state.source === 'threshold'")
        found = self.card()
        self.assertFalse(found["shown"])
        self.assertFalse(self.banner()["button"])
        self.assertEqual(found["expanded"], "false")

    def test_a_card_that_could_not_be_fetched_says_so(self):
        self.guide()
        self.script("""window.__fetch = window.fetch;
            window.fetch = (url, ...rest) => String(url).includes('/api/tree/mismatch') ? Promise.reject(new Error('down')) : window.__fetch(url, ...rest); return 1""")
        self.press()
        self.wait_for("document.querySelector('#mismatch-card .note') !== null")
        self.assertEqual(self.state("document.querySelector('#mismatch-card .note').textContent"), "The Viewer could not list them.")
        self.assertEqual(self.card()["boxes"], [])


if __name__ == "__main__":
    unittest.main()
