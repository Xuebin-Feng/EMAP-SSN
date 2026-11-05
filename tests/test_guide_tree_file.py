"""Where an alignment's guide tree lives, how the tree's labels are quoted, and how the file is read back (utilities/Guide_Tree_File.py)."""

import os
import pathlib
import re
import subprocess
import sys
import tempfile
import traceback
import unittest
from unittest import mock

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities import Guide_Tree_File
from utilities.Guide_Tree_File import (
    GUIDE_TREE_SUBFOLDER,
    GuideTreeError,
    guide_tree_path_for_alignment,
    newick_label,
    nexus_string,
    parse_guide_tree,
    read_guide_tree,
)


class GuideTreePathTests(unittest.TestCase):
    def test_the_subfolder_is_fixed(self):
        self.assertEqual(GUIDE_TREE_SUBFOLDER, "Guide_Trees")

    def test_a_fasta_alignment_has_its_tree_in_the_guide_trees_folder(self):
        folder = os.path.join(os.sep, "data", "msa")
        self.assertEqual(
            guide_tree_path_for_alignment(os.path.join(folder, "Set_[esm2]_alignment.fasta")),
            os.path.join(folder, "Guide_Trees", "Set_[esm2]_guide_tree.nex"),
        )

    def test_a_sparse_alignment_has_the_tree_of_its_fasta(self):
        folder = os.path.join(os.sep, "data", "msa")
        for name in ("Set_[esm2]_alignment_sparse.h5", "Set_[esm2]_alignment.h5"):
            with self.subTest(name=name):
                self.assertEqual(
                    guide_tree_path_for_alignment(os.path.join(folder, name)),
                    os.path.join(folder, "Guide_Trees", "Set_[esm2]_guide_tree.nex"),
                )

    def test_the_folder_keeps_its_spelling_relative_or_absolute(self):
        relative = os.path.join("Input_Files", "Multiple_Alignments", "A_alignment.fasta")
        self.assertEqual(
            guide_tree_path_for_alignment(relative),
            os.path.join("Input_Files", "Multiple_Alignments", "Guide_Trees", "A_guide_tree.nex"),
        )
        self.assertEqual(guide_tree_path_for_alignment("A_alignment.fasta"), os.path.join("Guide_Trees", "A_guide_tree.nex"))
        absolute = os.path.abspath(os.path.join("elsewhere", "A_alignment.fasta"))
        self.assertEqual(
            guide_tree_path_for_alignment(absolute),
            os.path.join(os.path.dirname(absolute), "Guide_Trees", "A_guide_tree.nex"),
        )

    def test_brackets_dots_and_a_repeated_alignment_word_stay_in_the_name(self):
        # Only the last "_alignment" is the tool's; the set name keeps everything before it.
        self.assertEqual(
            os.path.basename(guide_tree_path_for_alignment("my_alignment_set.v2_[m_1]_alignment.fasta")),
            "my_alignment_set.v2_[m_1]_guide_tree.nex",
        )

    def test_a_name_that_does_not_follow_the_tool_has_no_tree(self):
        for name in (
            "mafft_result.fasta",           # an external alignment
            "Set_[m]_alignment_edited.fasta",  # renamed
            "Set_[m]_alignment.fa",         # other extension
            "Set_[m]_alignment.fasta.bak",
            "Set_[m]_sparse.h5",            # sparse file without the alignment word
            "_alignment.fasta",             # no set name
            "alignment.fasta",
            "",
            "   ",
            None,
        ):
            with self.subTest(name=name):
                self.assertIsNone(guide_tree_path_for_alignment(name))

    def test_the_suffixes_are_read_without_regard_to_case(self):
        self.assertEqual(
            os.path.basename(guide_tree_path_for_alignment("Set_ALIGNMENT.FASTA")), "Set_guide_tree.nex"
        )

    def test_a_path_object_works(self):
        path = pathlib.Path("msa") / "Set_alignment.fasta"
        self.assertEqual(guide_tree_path_for_alignment(path), os.path.join("msa", "Guide_Trees", "Set_guide_tree.nex"))


class LabelQuotingTests(unittest.TestCase):
    def test_plain_labels_stay_bare_and_everything_else_is_quoted(self):
        self.assertEqual(newick_label("sp|P12345.1|ABC"), "sp|P12345.1|ABC")
        self.assertEqual(newick_label("a_b"), "'a_b'")
        self.assertEqual(newick_label("x y"), "'x y'")
        self.assertEqual(newick_label("O'Brien"), "'O''Brien'")
        self.assertEqual(newick_label("a,b"), "'a,b'")
        self.assertEqual(newick_label("é"), "'é'")
        self.assertEqual(nexus_string("it's"), "'it''s'")
        self.assertEqual(nexus_string(7), "'7'")

    def test_the_writer_uses_these_functions(self):
        sys.path.insert(0, str(SRC_DIR / "tools"))
        try:
            import Embedding_MSA
        finally:
            sys.path.remove(str(SRC_DIR / "tools"))
        self.assertIs(Embedding_MSA.newick_label, newick_label)
        self.assertIs(Embedding_MSA.nexus_string, nexus_string)



def writer_module():
    """Embedding_MSA, which writes the files the reader has to take back."""
    sys.path.insert(0, str(SRC_DIR / "tools"))
    try:
        import Embedding_MSA
    finally:
        sys.path.remove(str(SRC_DIR / "tools"))
    return Embedding_MSA


# Headers the writer has to quote, and a reader has to unquote: a comma, a space, apostrophes (one doubled),
# underscores, brackets, semicolons, equals signs, colons, parentheses, non-ASCII text and padding.
AWKWARD_HEADERS = [
    "sp|P12345|NAME_HUMAN", "x y", "O'Brien", "pdb|1IYE|Chains_A,_B,_C", "Ünï;cödé [x] = y",
    "a(b):c;d", "tr|A0A1B2.1|Q9", "  leading and trailing  ", "日本語 ヘッダー", "it''s",
]


def leaf_path_lengths(parent, length, leaves):
    """The path length between every two of the nodes ``leaves``, from parent and branch-length arrays."""
    def chain(node):
        found = []
        while node >= 0:
            found.append(int(node))
            node = parent[node]
        return found

    depth = {}

    def depth_of(node):
        # The numbering is post-order, so a node's parent has a higher number: go from the root down.
        if node not in depth:
            above = chain(node)
            total = 0.0
            for step in reversed(above):
                if parent[step] >= 0:
                    total += float(length[step])
                depth[step] = total
        return depth[node]

    chains = [chain(leaf) for leaf in leaves]
    paths = np.zeros((len(leaves), len(leaves)))
    for i, first in enumerate(leaves):
        above = set(chains[i])
        for j, second in enumerate(leaves):
            meet = next(node for node in chains[j] if node in above)
            paths[i, j] = depth_of(int(first)) + depth_of(int(second)) - 2 * depth_of(meet)
    return paths


def without_quoted_words(text):
    """A Newick text with each quoted word replaced by '': what is left holds only its own punctuation."""
    return re.sub(r"'(?:[^']|'')*'", "''", text)


class WrittenTreeReadBackTests(unittest.TestCase):
    """What the writer saves is what the reader gets: headers, shape, lengths, labels and notes."""

    @classmethod
    def setUpClass(cls):
        cls.writer = writer_module()

    def write_and_read(self, headers, link, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "tree.nex"
            self.writer.write_guide_tree(path, link, headers, **kwargs)
            return read_guide_tree(path), path.read_text(encoding="utf-8")

    @staticmethod
    def upgma(count, seed=3):
        return sch.linkage(np.random.default_rng(seed).random((count, 3)), method="average")

    def test_every_awkward_header_comes_back_as_it_went_in(self):
        tree, _ = self.write_and_read(AWKWARD_HEADERS, self.upgma(len(AWKWARD_HEADERS)))
        self.assertEqual(sorted(tree.labels), sorted(AWKWARD_HEADERS))
        self.assertEqual((tree.node_count, tree.leaf_count), (2 * len(AWKWARD_HEADERS) - 1, len(AWKWARD_HEADERS)))

    def test_an_upgma_tree_is_rooted_and_its_paths_are_the_merge_heights(self):
        count = len(AWKWARD_HEADERS)
        link = self.upgma(count)
        tree, _ = self.write_and_read(AWKWARD_HEADERS, link)
        self.assertIs(tree.rooted, True)
        self.assertTrue(tree.has_lengths)
        self.assertEqual(tree.name, "guide_tree")
        order = [AWKWARD_HEADERS.index(label) for label in tree.labels]
        paths = leaf_path_lengths(tree.parent, tree.length, tree.leaf_nodes())
        expected = squareform(sch.cophenet(link))[np.ix_(order, order)]
        # The file keeps six significant digits of each length.
        np.testing.assert_allclose(paths, expected, rtol=1e-4, atol=1e-6)

    def test_a_neighbor_joining_tree_is_unrooted_and_keeps_its_path_lengths(self):
        count = len(AWKWARD_HEADERS)
        condensed = np.random.default_rng(5).random(count * (count - 1) // 2)
        link, lengths = self.writer.neighbor_joining_condensed(condensed, count, return_branch_lengths=True)
        tree, _ = self.write_and_read(AWKWARD_HEADERS, link, nj_branch_lengths=lengths)
        self.assertIs(tree.rooted, False)
        parent, length = self.writer.guide_tree_branches(link, count, lengths)
        # Midpoint rooting moves the root and keeps every path between two leaves.
        expected = leaf_path_lengths(parent, length, list(range(count)))
        order = [AWKWARD_HEADERS.index(label) for label in tree.labels]
        paths = leaf_path_lengths(tree.parent, tree.length, tree.leaf_nodes())
        np.testing.assert_allclose(paths, expected[np.ix_(order, order)], rtol=1e-4, atol=1e-5)

    def test_the_numbering_is_post_order_and_the_leaves_run_in_the_files_order(self):
        tree, text = self.write_and_read(AWKWARD_HEADERS, self.upgma(len(AWKWARD_HEADERS), seed=9))
        parent = tree.parent
        self.assertEqual(int(parent[-1]), -1)
        self.assertTrue(np.all(parent[:-1] > np.arange(tree.node_count - 1)))
        self.assertEqual(np.count_nonzero(parent < 0), 1)
        leaves = tree.leaf_nodes()
        self.assertEqual(len(leaves), tree.leaf_count)
        self.assertTrue(np.all(np.diff(leaves) > 0))
        line = next(row for row in text.splitlines() if row.lstrip().startswith("TREE "))
        expected = sorted(AWKWARD_HEADERS, key=lambda header: line.index(newick_label(header)))
        self.assertEqual(tree.labels, expected)

    def test_split_support_labels_come_back_on_their_internal_nodes(self):
        writer, count = self.writer, 8
        headers = [f"seq {i}" for i in range(count)]
        first, second = self.upgma(count, seed=1), self.upgma(count, seed=2)
        keys = writer.guide_tree_leaf_keys(count, 42)
        replicate_keys = np.sort(np.concatenate([writer.replicate_split_keys(t, count, keys) for t in (first, second)]))
        tree, text = self.write_and_read(
            headers, first, leaf_keys=keys, replicate_keys=replicate_keys, num_trees=2)
        line = next(row for row in text.splitlines() if row.lstrip().startswith("TREE "))
        # The ")" of the text come in the order the reader numbers the internal nodes.
        written = [int(value) if value else -1 for value in re.findall(r"\)(\d*)", without_quoted_words(line))]
        internal = np.flatnonzero(np.isin(np.arange(tree.node_count), tree.leaf_nodes(), invert=True))
        self.assertEqual(list(tree.support[internal]), written)
        self.assertTrue(any(value >= 0 for value in written))
        self.assertTrue(all(value == -1 for value in tree.support[tree.leaf_nodes()]))
        self.assertIn("not bootstrap", tree.notes["INTERNAL_LABELS"])

    def test_the_run_notes_come_back_with_their_quotes(self):
        notes = [("ALIGNMENT", "Set_[E1_RA]_alignment.fasta"), ("ODD", "O'Brien; [x] = y"), ("NUM_TREES", 7)]
        tree, _ = self.write_and_read(AWKWARD_HEADERS, self.upgma(len(AWKWARD_HEADERS)), notes=notes)
        self.assertEqual(tree.notes["ALIGNMENT"], "Set_[E1_RA]_alignment.fasta")
        self.assertEqual(tree.notes["ODD"], "O'Brien; [x] = y")
        self.assertEqual(tree.notes["NUM_TREES"], "7")
        self.assertEqual(tree.notes["SEQUENCES"], str(len(AWKWARD_HEADERS)))

    def test_one_and_two_sequences(self):
        tree, _ = self.write_and_read(["only one"], np.zeros((0, 4)))
        self.assertEqual((tree.labels, list(tree.parent), tree.node_count), (["only one"], [-1], 1))
        tree, _ = self.write_and_read(["a b", "c,d"], self.upgma(2))
        self.assertEqual(sorted(tree.labels), ["a b", "c,d"])
        self.assertEqual(list(tree.parent), [2, 2, -1])

    def test_a_deep_tree_reads_without_recursion(self):
        count = 20000
        link = np.zeros((count - 1, 4))
        link[0] = [0, 1, 1.0, 2]
        for step in range(1, count - 1):
            link[step] = [step + 1, count + step - 1, step + 1.0, step + 2]
        tree, _ = self.write_and_read([f"s{i}" for i in range(count)], link)
        self.assertEqual((tree.leaf_count, tree.node_count), (count, 2 * count - 1))

    def test_a_tree_of_thousands_of_leaves_comes_back_whole(self):
        count = 5000
        link = sch.linkage(np.random.default_rng(1).random((count, 3)), method="average")
        headers = [f"sp|Q{i:05d}|PROTEIN_{i} some name, organism {i}" for i in range(count)]
        tree, _ = self.write_and_read(headers, link)
        self.assertEqual(sorted(tree.labels), sorted(headers))


def nexus_text(tree="(a:1,b:2);", flag="[&R]", notes="", block="TREES", name="t"):
    """A NEXUS file text with one TREE command."""
    return f"#NEXUS\n{notes}BEGIN {block};\n    TREE {name} = {flag} {tree}\nEND;\n"


class ParserTests(unittest.TestCase):
    """The reader on hand-written NEXUS, which the writer does not make."""

    def test_the_rooting_flag_and_the_trees_name(self):
        self.assertIs(parse_guide_tree(nexus_text(flag="[&R]")).rooted, True)
        self.assertIs(parse_guide_tree(nexus_text(flag="[&U]")).rooted, False)
        self.assertIs(parse_guide_tree(nexus_text(flag="[ & r ]")).rooted, True)
        self.assertIsNone(parse_guide_tree(nexus_text(flag="")).rooted)
        self.assertIsNone(parse_guide_tree(nexus_text(flag="[&W 0.5]")).rooted)
        self.assertEqual(parse_guide_tree(nexus_text(name="* best one")).name, "best one")

    def test_only_a_flag_before_the_tree_says_whether_it_is_rooted(self):
        # Metacomments inside the tree text (a node's own annotation) say nothing of its rooting.
        self.assertIs(parse_guide_tree(nexus_text("(a[&U]:1,b[&U]:2)[&U];", flag="[&R]")).rooted, True)
        self.assertIs(parse_guide_tree(nexus_text("(a[&R]:1,b:2);", flag="[&U]")).rooted, False)
        self.assertIsNone(parse_guide_tree(nexus_text("(a[&R]:1,b:2);", flag="")).rooted)

    def test_keywords_are_read_in_any_case_and_a_block_may_end_with_endblock(self):
        text = "#nexus\nbegin trees;\n  tree t = (a,b);\nendblock;\n"
        self.assertEqual(parse_guide_tree(text).labels, ["a", "b"])
        taxa = "#NEXUS\nBEGIN TAXA; DIMENSIONS NTAX=2; END;\nBEGIN TREES; UTREE u = (a,b); END;"
        self.assertEqual(parse_guide_tree(taxa).name, "u")

    def test_an_unquoted_underscore_is_a_space_and_a_quoted_one_stays(self):
        self.assertEqual(
            parse_guide_tree(nexus_text("(sp_one,'sp_two',sp|P1.1);")).labels, ["sp one", "sp_two", "sp|P1.1"])

    def test_a_doubled_quote_is_one_and_a_quoted_word_may_hold_punctuation(self):
        tree = parse_guide_tree(nexus_text("('O''Brien','x(1):2;3,4[5]=6');"))
        self.assertEqual(tree.labels, ["O'Brien", "x(1):2;3,4[5]=6"])

    def test_comments_may_nest_and_sit_between_any_two_tokens(self):
        tree = parse_guide_tree(nexus_text("[lead](a[c1 [nested] c2]:[x]1,[y]b:2[z])[end]:0.5;"))
        self.assertEqual(tree.labels, ["a", "b"])
        np.testing.assert_allclose(tree.length, [1.0, 2.0, 0.5])

    def test_lengths_support_and_the_root(self):
        tree = parse_guide_tree(nexus_text("((a:1,b:2.5e-1)87:0.5,c:3)99;"))
        self.assertEqual(tree.labels, ["a", "b", "c"])
        # Post-order: a, b, the clade (a,b), c, the root.
        self.assertEqual(list(tree.parent), [2, 2, 4, 4, -1])
        np.testing.assert_allclose(tree.length, [1.0, 0.25, 0.5, 3.0, 0.0])
        self.assertEqual(list(tree.support), [-1, -1, 87, -1, 99])
        self.assertTrue(tree.has_lengths)

    def test_a_tree_without_lengths(self):
        tree = parse_guide_tree(nexus_text("((a,b),c);"))
        self.assertFalse(tree.has_lengths)
        self.assertEqual(list(tree.length), [0.0] * 5)

    def test_a_label_that_is_not_a_percentage_is_not_support(self):
        tree = parse_guide_tree(nexus_text("((a,b)x,(c,d)101,(e,f)-5,(g,h)0);"))
        leaves = set(tree.leaf_nodes())
        internal = [index for index in range(tree.node_count) if index not in leaves]
        # Of the four clades and the root, only the label 0 is a percentage.
        self.assertEqual(sorted(int(tree.support[index]) for index in internal), [-1, -1, -1, -1, 0])

    def test_notes_come_from_the_emapssn_block_only(self):
        notes = ("BEGIN EMAPSSN;\n  [a comment; with = signs]\n  ALIGNMENT = 'it''s.fasta';\n  ALIGNMENT = 'second';\n"
                 "  SEQUENCES = '2';\n  BARE = word;\n  BROKEN 'x';\nEND;\nBEGIN OTHER;\n  LEAKED = 'no';\nEND;\n")
        tree = parse_guide_tree(nexus_text(notes=notes))
        self.assertEqual(tree.notes, {"ALIGNMENT": "it's.fasta", "SEQUENCES": "2", "BARE": "word"})

    def test_only_the_first_tree_is_taken(self):
        text = "#NEXUS\nBEGIN TREES;\n  TREE one = (a,b);\n  TREE two = (c,d,e);\nEND;\n"
        tree = parse_guide_tree(text)
        self.assertEqual((tree.name, tree.labels), ("one", ["a", "b"]))

    def test_a_single_leaf_is_a_tree(self):
        tree = parse_guide_tree(nexus_text("'only one';"))
        self.assertEqual((tree.labels, list(tree.parent), tree.node_count), (["only one"], [-1], 1))

    def test_a_node_may_have_one_child_or_many(self):
        tree = parse_guide_tree(nexus_text("((a),(b,c,d,e));"))
        self.assertEqual(tree.labels, ["a", "b", "c", "d", "e"])
        # a, (a), b, c, d, e, (b,c,d,e), root.
        self.assertEqual(list(tree.parent), [1, 7, 6, 6, 6, 6, 7, -1])

    def test_text_that_is_not_a_guide_tree_is_refused_with_a_reason(self):
        cases = {
            "no #NEXUS": "BEGIN TREES; TREE t = (a,b); END;",
            "empty": "",
            "only the header": "#NEXUS\n",
            "no TREES block": "#NEXUS\nBEGIN EMAPSSN; X = 'y'; END;\n",
            "TRANSLATE table": "#NEXUS\nBEGIN TREES; TRANSLATE 1 a, 2 b; TREE t = (1,2); END;\n",
            "no semicolon": "#NEXUS\nBEGIN TREES; TREE t = (a,b)",
            "open parenthesis": nexus_text("((a,b);"),
            "extra parenthesis": nexus_text("(a,b));"),
            "comma outside": nexus_text("a,b;"),
            "empty parentheses": nexus_text("();"),
            "empty leaf": nexus_text("(a,);"),
            "leaf without a name": nexus_text("(:1,b);"),
            "two names": nexus_text("(a b,c);"),
            "label after a length": nexus_text("(a:1 b,c);"),
            "two lengths": nexus_text("(a:1:2,b);"),
            "text after the root": nexus_text("(a,b)(c,d);"),
            "not a number": nexus_text("(a:xyz,b);"),
            "not finite": nexus_text("(a:nan,b);"),
            "infinite": nexus_text("(a:inf,b);"),
            "length at the end": "#NEXUS\nBEGIN TREES; TREE t = (a:",
            "equals inside": nexus_text("(a=b,c);"),
            "unclosed quote": nexus_text("('a,b);"),
            "stray bracket": nexus_text("(a],b);"),
            "unclosed comment": nexus_text("(a[x,b);"),
            "no tree in the command": "#NEXUS\nBEGIN TREES; TREE t; END;\n",
            "empty tree": nexus_text(";"),
        }
        for label, text in cases.items():
            with self.subTest(label):
                with self.assertRaises(GuideTreeError) as raised:
                    parse_guide_tree(text)
                self.assertTrue(str(raised.exception))

    def test_the_limits_are_enforced(self):
        with self.assertRaisesRegex(GuideTreeError, "more than 3 leaves"):
            parse_guide_tree(nexus_text("((a,b),(c,d));"), max_leaves=3)
        parse_guide_tree(nexus_text("((a,b),(c,d));"), max_leaves=4)
        with self.assertRaisesRegex(GuideTreeError, "more than 20 nested nodes"):
            parse_guide_tree(nexus_text("(" * 100 + "a" + ",b)" * 100 + ";"), max_leaves=10)
        # Few leaves, shallow, but more nodes than twice the leaves allowed: each leaf wrapped in three single-child nodes.
        wrapped = "(" + ",".join(f"(((x{i})))" for i in range(10)) + ");"
        with self.assertRaisesRegex(GuideTreeError, "more than 20 nodes"):
            parse_guide_tree(nexus_text(wrapped), max_leaves=10)
        parse_guide_tree(nexus_text(wrapped), max_leaves=21)
        with mock.patch.object(Guide_Tree_File, "MAX_LABEL_CHARACTERS", 5):
            with self.assertRaisesRegex(GuideTreeError, "longer than 5 characters"):
                parse_guide_tree(nexus_text("(abcdef,b);"))
            parse_guide_tree(nexus_text("(abcde,b);"))

    def test_a_flood_of_open_parentheses_is_refused_before_it_fills_memory(self):
        # Refused for what it is while the text is still being read, not for its missing end: nothing past the
        # limit is kept.
        with self.assertRaisesRegex(GuideTreeError, "more than 200 nested nodes"):
            parse_guide_tree(nexus_text("(" * 5000), max_leaves=100)

    def test_a_deep_text_is_read_without_recursion(self):
        depth = 50000
        text = nexus_text("(" * depth + "a" + ",b)" * depth + ";")
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(len(traceback.extract_stack()) + 80)
        try:
            tree = parse_guide_tree(text)
        finally:
            sys.setrecursionlimit(limit)
        self.assertEqual((tree.leaf_count, tree.node_count), (depth + 1, 2 * depth + 1))


class FileReadTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = pathlib.Path(folder.name)

    def write(self, data, name="tree.nex"):
        path = self.folder / name
        path.write_bytes(data)
        return path

    def test_a_file_is_read_as_utf8_with_or_without_a_byte_order_mark(self):
        text = nexus_text("('Ünï',b);").encode("utf-8")
        self.assertEqual(read_guide_tree(self.write(text)).labels, ["Ünï", "b"])
        self.assertEqual(read_guide_tree(self.write(b"\xef\xbb\xbf" + text)).labels, ["Ünï", "b"])

    def test_a_file_that_is_not_utf8_is_refused(self):
        with self.assertRaisesRegex(GuideTreeError, "UTF-8"):
            read_guide_tree(self.write(nexus_text("('\xdc',b);").encode("latin-1")))

    def test_a_file_over_the_limit_is_refused_unread(self):
        path = self.write(nexus_text().encode("utf-8"))
        with self.assertRaisesRegex(GuideTreeError, "larger"):
            read_guide_tree(path, max_bytes=10)
        # Refused by its size, before any of it is read into memory.
        reads = []

        class Spy:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *error):
                self.handle.close()

            def fileno(self):
                return self.handle.fileno()

            def read(self, size=-1):
                reads.append(size)
                return self.handle.read(size)

        with mock.patch.object(Guide_Tree_File, "open", lambda name, mode: Spy(open(name, mode)), create=True):
            with self.assertRaisesRegex(GuideTreeError, "larger"):
                read_guide_tree(path, max_bytes=10)
        self.assertEqual(reads, [])

    def test_a_missing_file_is_the_callers_oserror(self):
        with self.assertRaises(FileNotFoundError):
            read_guide_tree(self.folder / "absent.nex")
        with self.assertRaises(OSError):
            read_guide_tree(self.folder)

class LightImportTests(unittest.TestCase):
    def test_the_module_loads_no_heavy_library(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "import utilities.Guide_Tree_File;"
            "heavy = [m for m in ('torch', 'numba', 'PySide6', 'numpy', 'scipy') if m in sys.modules];"
            "print(','.join(heavy))"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code, str(SRC_DIR)],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
