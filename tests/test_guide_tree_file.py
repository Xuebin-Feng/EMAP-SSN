"""Where an alignment's guide tree lives, and how the tree's labels are quoted (utilities/Guide_Tree_File.py)."""

import os
import pathlib
import subprocess
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities import Guide_Tree_File
from utilities.Guide_Tree_File import (
    GUIDE_TREE_SUBFOLDER,
    guide_tree_path_for_alignment,
    newick_label,
    nexus_string,
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
