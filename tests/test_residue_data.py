"""Per-residue data from the loaded alignment (utilities/Residue_Data.py).

Covers the two properties `esmfold` writes beside a structure, information
content and residue agreement, on hand-made alignments (gaps, non-standard
symbols, a node the alignment lacks, a sequence it holds differently, a single
row, leading and trailing gaps), their agreement with the `logo` command, the
registry the properties come from, and the cost of an alignment of many rows.
"""

import hashlib
import json
import math
import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy import sparse


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands import logo
from tests.sparse_alignment import sparse_alignment
from utilities import Alignment_Profiles, Residue_Data
from utilities.Sequence_Utils import INT_TO_AA

INT_TO_AA_CODE = {letter: code for code, letter in INT_TO_AA.items()}


def inputs_for(records, msa_file="C:/data/family.fasta", mapping=None):
    """ViewerInputs for (header, aligned sequence) rows loaded as the Viewer loads them; node i is row i."""
    aln = sparse_alignment(records)
    alignment = SimpleNamespace(aln=aln, msa_file=msa_file)
    mapping = np.arange(len(records)) if mapping is None else np.asarray(mapping)
    return Residue_Data.ViewerInputs(alignment, mapping, tuple(header for header, _ in records))


def ungapped(aligned):
    return "".join(character for character in aligned if character not in "-.")


# Four rows over eight columns. Node 0 has a gap in column 3, node 1 an X in column 6.
#
#   col   0 1 2 3 4 5 6 7
#   n0    A C D - E F G H
#   n1    A C E K E F X H
#   n2    A S D - E - G -
#   n3    G C D K Q F - H
HAND_ROWS = [
    ("n0", "ACD-EFGH"),
    ("n1", "ACEKEFXH"),
    ("n2", "ASD-E-G-"),
    ("n3", "GCDKQF-H"),
]


class SameResidueTests(unittest.TestCase):
    def values(self, inputs, node, sequence=None, property_id="same_residue_pct"):
        sequence = ungapped(HAND_ROWS[node][1]) if sequence is None else sequence
        return Residue_Data.PROVIDERS[property_id](inputs, node, sequence)

    def test_hand_made_alignment_counts_standard_residues_with_the_node_included(self):
        inputs = inputs_for(HAND_ROWS)
        # n0: column 0 has A A A G (3 of 4), 1 C C S C (3 of 4), 2 D E D D (3 of 4), 4 E E E Q (3 of 4),
        # 5 F F - F (3 of 3), 6 G X G - (2 of 2: X and the gap are not counted), 7 H H - H (3 of 3).
        data = self.values(inputs, 0)
        self.assertEqual(data["values"], [75.0, 75.0, 75.0, 75.0, 100.0, 100.0, 100.0])
        self.assertEqual(data["source"]["columns"], [0, 1, 2, 4, 5, 6, 7])

    def test_a_node_with_a_non_standard_symbol_has_no_value_there(self):
        inputs = inputs_for(HAND_ROWS)
        # n1 is A C E K E F X H: column 2 E is 1 of 4 (D E D D), column 3 K is 2 of 2 (the others are gaps).
        data = self.values(inputs, 1)
        self.assertEqual(data["values"], [75.0, 75.0, 25.0, 100.0, 75.0, 100.0, None, 100.0])
        # The symbol still has its column, which is how the residue stays in step with the structure.
        self.assertEqual(data["source"]["columns"], [0, 1, 2, 3, 4, 5, 6, 7])

    def test_a_single_row_agrees_with_itself_everywhere(self):
        inputs = inputs_for([("only", "AC-DE")])
        data = Residue_Data.same_residue_pct(inputs, 0, "ACDE")
        self.assertEqual(data["values"], [100.0] * 4)
        self.assertEqual(data["source"]["sequences"], 1)

    def test_values_are_rounded_to_two_decimals(self):
        rows = [("a", "A"), ("b", "G"), ("c", "S")]
        data = Residue_Data.same_residue_pct(inputs_for(rows), 0, "A")
        self.assertEqual(data["values"], [33.33])

    def test_leading_and_trailing_gaps_map_residues_to_their_columns(self):
        rows = [("n0", "--ACD--"), ("n1", "GGACDGG"), ("n2", "-SA-D--")]
        inputs = inputs_for(rows)
        data = Residue_Data.same_residue_pct(inputs, 0, "ACD")
        self.assertEqual(data["source"]["columns"], [2, 3, 4])
        # Column 2: A A A; column 3: C C (n2 has a gap); column 4: D D D.
        self.assertEqual(data["values"], [100.0, 100.0, 100.0])
        trailing = Residue_Data.same_residue_pct(inputs, 1, "GGACDGG")
        self.assertEqual(trailing["source"]["columns"], [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(trailing["values"], [100.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0])

    def test_the_sequence_may_be_in_any_case(self):
        data = Residue_Data.same_residue_pct(inputs_for(HAND_ROWS), 0, "acdefgh")
        self.assertEqual(len(data["values"]), 7)

    def test_each_reason_for_no_data_is_its_own_kind(self):
        inputs = inputs_for(HAND_ROWS)
        with self.assertRaises(Residue_Data.ResidueDataUnavailable) as raised:
            Residue_Data.same_residue_pct(Residue_Data.ViewerInputs(), 0, "ACDEFGH")
        self.assertEqual(raised.exception.kind, Residue_Data.NO_ALIGNMENT)
        # An alignment object that holds no rows is no alignment either.
        empty = Residue_Data.ViewerInputs(SimpleNamespace(aln=None, msa_file="x"), np.array([0]), ())
        with self.assertRaises(Residue_Data.ResidueDataUnavailable) as raised:
            Residue_Data.same_residue_pct(empty, 0, "ACDEFGH")
        self.assertEqual(raised.exception.kind, Residue_Data.NO_ALIGNMENT)

        absent = Residue_Data.ViewerInputs(inputs.alignment, np.array([0, -1, 2, 3]), inputs.full_headers)
        for node in (1, 9, -1):
            with self.subTest(node=node), self.assertRaises(Residue_Data.ResidueDataUnavailable) as raised:
                Residue_Data.same_residue_pct(absent, node, "ACDEFGH")
            self.assertEqual(raised.exception.kind, Residue_Data.NOT_IN_ALIGNMENT)

        for label, sequence in (
            ("a residue changed", "ACDEFGA"),
            ("a residue missing", "ACDEFG"),
            ("a residue added", "ACDEFGHK"),
            ("another protein", "MKVL"),
        ):
            with self.subTest(label), self.assertRaises(Residue_Data.ResidueDataUnavailable) as raised:
                Residue_Data.same_residue_pct(inputs, 0, sequence)
            self.assertEqual(raised.exception.kind, Residue_Data.SEQUENCE_DIFFERS)
            self.assertIn("differs", raised.exception.reason)

    def test_the_node_is_found_through_the_viewers_mapping_not_its_own_position(self):
        # Viewer node 0 is row 3 of the alignment.
        inputs = inputs_for(HAND_ROWS, mapping=[3, 2, 1, 0])
        data = Residue_Data.same_residue_pct(inputs, 0, ungapped(HAND_ROWS[3][1]))
        self.assertEqual(data["source"]["columns"], [0, 1, 2, 3, 4, 5, 7])
        self.assertEqual(data["values"][0], 25.0)  # The G of column 0 among A A A G.

    def test_a_matrix_that_is_dense_or_column_ordered_gives_the_same_values(self):
        loaded = inputs_for(HAND_ROWS)
        dense = loaded.alignment.aln.matrix.toarray()
        for label, matrix in (("dense", dense), ("csc", sparse.csc_matrix(dense)), ("coo", sparse.coo_matrix(dense))):
            with self.subTest(label):
                aln = SimpleNamespace(matrix=matrix, int_to_aa=dict(INT_TO_AA), headers=[h for h, _ in HAND_ROWS])
                stand_in = Residue_Data.ViewerInputs(
                    SimpleNamespace(aln=aln, msa_file="a.fasta"), np.arange(4), loaded.full_headers
                )
                for node in range(4):
                    sequence = ungapped(HAND_ROWS[node][1])
                    self.assertEqual(
                        Residue_Data.same_residue_pct(stand_in, node, sequence)["values"],
                        Residue_Data.same_residue_pct(loaded, node, sequence)["values"],
                    )

    def test_the_description_says_what_is_counted_and_that_it_is_this_proteins_residue(self):
        description = Residue_Data.same_residue_pct(inputs_for(HAND_ROWS), 0, "ACDEFGH")["description"]
        for text in ("standard amino acid", "same amino acid as this protein", "own row is", "specific to this protein"):
            self.assertIn(text, description)


class InformationContentTests(unittest.TestCase):
    def conserved_column(self, rows, aligned_rows=None):
        """A column of `rows` aligned rows holding A (or a gap), after a tag that keeps every row distinct."""
        records = []
        for index in range(rows):
            tag = "".join("ACDEFGHIKLMNPQRSTVWY"[(index // 20 ** k) % 20] for k in range(3))
            records.append((f"s{index}", tag + ("A" if aligned_rows is None or index < aligned_rows else "-")))
        return records

    @staticmethod
    def expected(n, total_rows, probabilities):
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        correction = 19.0 / (2.0 * math.log(2) * n)
        information = max(0.0, math.log2(20) - (entropy + correction))
        return information * (n / total_rows) / math.log2(20) * 100.0

    def test_a_fully_conserved_column_from_first_principles(self):
        records = self.conserved_column(40)
        data = Residue_Data.information_content(inputs_for(records), 0, ungapped(records[0][1]))
        self.assertEqual(data["values"][-1], round(self.expected(40, 40, [1.0]), 2))
        self.assertAlmostEqual(data["values"][-1], 92.07, places=2)

    def test_gaps_scale_the_height_by_occupancy(self):
        records = self.conserved_column(40, aligned_rows=20)
        data = Residue_Data.information_content(inputs_for(records), 0, ungapped(records[0][1]))
        self.assertEqual(data["values"][-1], round(self.expected(20, 40, [1.0]), 2))
        self.assertAlmostEqual(data["values"][-1], 42.07, places=2)

    def test_a_column_with_few_sequences_floors_at_zero(self):
        # Two sequences carry too little information to beat the small-sample correction.
        records = self.conserved_column(4, aligned_rows=2)
        data = Residue_Data.information_content(inputs_for(records), 0, ungapped(records[0][1]))
        self.assertEqual(data["values"][-1], 0.0)

    def test_a_column_is_one_value_for_every_protein_with_a_residue_there(self):
        inputs = inputs_for(HAND_ROWS)
        first = Residue_Data.information_content(inputs, 0, "ACDEFGH")
        second = Residue_Data.information_content(inputs, 3, ungapped(HAND_ROWS[3][1]))
        columns = dict(zip(first["source"]["columns"], first["values"]))
        for column, value in zip(second["source"]["columns"], second["values"]):
            if column in columns:
                self.assertEqual(value, columns[column])

    def test_the_values_are_the_logo_commands_column_heights_as_a_percentage_of_the_maximum(self):
        rng = np.random.default_rng(42)
        alphabet = list("ACDEFGHIKLMNPQRSTVWY") * 2 + ["-"] * 7 + ["X"]
        records, seen = [], set()
        while len(records) < 36:
            row = "".join(rng.choice(alphabet, size=14))
            if row not in seen:
                seen.add(row)
                records.append((f"s{len(records)}", row))
        inputs = inputs_for(records)
        strings = Alignment_Profiles.aligned_row_strings(inputs.alignment.aln, list(range(len(records))))
        for node in (0, 5, 35):
            sequence = ungapped(records[node][1])
            data = Residue_Data.information_content(inputs, node, sequence)
            for column, value in zip(data["source"]["columns"], data["values"]):
                matrix, _ = logo.calculate_logo_matrix(strings, [column], mode="bits", gap_mode="with_gap")
                expected = round(float(matrix[0].sum()) / math.log2(20) * 100.0, 2)
                self.assertEqual(value, expected, (node, column))

    def test_source_names_the_measure_and_its_maximum(self):
        data = Residue_Data.information_content(inputs_for(HAND_ROWS), 0, "ACDEFGH")
        self.assertEqual(data["source"]["measure"], "logo bits, with_gap, small-sample corrected")
        self.assertEqual(data["source"]["max_bits"], 4.3219)
        self.assertEqual((data["label"], data["unit"], data["range"]), ("Information Content", "%", [0, 100]))
        self.assertTrue(all(0.0 <= value <= 100.0 for value in data["values"]))

    def test_the_same_guard_and_mapping_apply(self):
        inputs = inputs_for(HAND_ROWS)
        with self.assertRaises(Residue_Data.ResidueDataUnavailable) as raised:
            Residue_Data.information_content(inputs, 0, "ACDEFGA")
        self.assertEqual(raised.exception.kind, Residue_Data.SEQUENCE_DIFFERS)
        with self.assertRaises(Residue_Data.ResidueDataUnavailable) as raised:
            Residue_Data.information_content(Residue_Data.ViewerInputs(), 0, "ACDEFGH")
        self.assertEqual(raised.exception.kind, Residue_Data.NO_ALIGNMENT)


class SourceTests(unittest.TestCase):
    def test_the_source_describes_the_alignment_the_values_came_from(self):
        data = Residue_Data.same_residue_pct(inputs_for(HAND_ROWS, msa_file="C:/data/family.fasta"), 0, "ACDEFGH")
        source = data["source"]
        self.assertEqual(source["kind"], "msa")
        self.assertEqual(source["file"], "family.fasta")
        self.assertEqual(source["sequences"], 4)
        self.assertRegex(source["alignment_id"], r"^sha256:[0-9a-f]{64}$")

    def test_the_alignment_id_is_a_content_hash_that_follows_the_loaded_rows(self):
        def identity(rows, **options):
            return Residue_Data.same_residue_pct(inputs_for(rows, **options), 0, ungapped(rows[0][1]))["source"]["alignment_id"]

        first = identity(HAND_ROWS)
        # Loaded again, from another place: the same rows, the same identity.
        self.assertEqual(identity(HAND_ROWS, msa_file="D:/elsewhere/copy.fasta"), first)
        # One residue different, or one row fewer, is another alignment.
        changed = [("n0", "ACD-EFGA")] + HAND_ROWS[1:]
        self.assertNotEqual(identity(changed), first)
        self.assertNotEqual(identity(HAND_ROWS[:3]), first)
        # The digest is of the matrix, headers and symbol table: recompute it independently.
        aln = sparse_alignment(HAND_ROWS)
        matrix = aln.matrix.tocsr()
        matrix.sort_indices()
        digest = hashlib.sha256(b"emapssn-alignment-1\n")
        digest.update(b"4x8\n")
        for header, _ in HAND_ROWS:
            digest.update(header.encode() + b"\n")
        digest.update(json.dumps({str(code): symbol for code, symbol in sorted(aln.int_to_aa.items())}).encode())
        digest.update(matrix.indptr.astype("<i8").tobytes())
        digest.update(matrix.indices.astype("<i4").tobytes())
        digest.update(matrix.data.astype("<u1").tobytes())
        self.assertEqual(first, "sha256:" + digest.hexdigest())

    def test_both_properties_of_a_node_share_the_columns_and_the_identity(self):
        inputs = inputs_for(HAND_ROWS)
        information = Residue_Data.information_content(inputs, 0, "ACDEFGH")["source"]
        agreement = Residue_Data.same_residue_pct(inputs, 0, "ACDEFGH")["source"]
        for key in ("kind", "file", "alignment_id", "columns", "sequences"):
            self.assertEqual(information[key], agreement[key])


class RegistryTests(unittest.TestCase):
    def test_the_properties_come_in_the_order_the_page_lists_them(self):
        self.assertEqual(list(Residue_Data.PROVIDERS), ["information_content", "same_residue_pct"])
        result = Residue_Data.compute_residue_data(inputs_for(HAND_ROWS), 0, "ACDEFGH")
        self.assertEqual(list(result.properties), ["information_content", "same_residue_pct"])
        self.assertEqual(result.unavailable, {})
        for property_id, data in result.properties.items():
            self.assertEqual(
                sorted(data), ["description", "label", "range", "source", "unit", "values"], property_id
            )
            self.assertEqual(len(data["values"]), 7)

    def test_the_fold_view_page_translates_the_labels_the_providers_write(self):
        # The labels are saved in the file in English; the page shows them in its language by this table.
        page = os.path.join(SRC_DIR, "web_ui", "esmfold_layers.js")
        with open(page, encoding="utf-8") as handle:
            source = handle.read()
        for property_id, provider in Residue_Data.PROVIDERS.items():
            label = provider(inputs_for(HAND_ROWS), 0, "ACDEFGH")["label"]
            self.assertIn(f"['{label}', t(\"{label}\")]", source, property_id)

    def test_without_an_alignment_every_property_is_unavailable_for_the_same_reason(self):
        result = Residue_Data.compute_residue_data(Residue_Data.ViewerInputs(), 0, "ACDE")
        self.assertEqual(result.properties, {})
        self.assertEqual(
            {property_id: error.kind for property_id, error in result.unavailable.items()},
            {"information_content": Residue_Data.NO_ALIGNMENT, "same_residue_pct": Residue_Data.NO_ALIGNMENT},
        )

    def test_a_new_property_is_one_function_and_one_entry(self):
        def hydrophobic(inputs, node_index, sequence):
            return {"label": "Hydrophobic", "unit": "", "range": [0, 1], "description": "A or V.",
                    "values": [1 if letter in "AV" else 0 for letter in sequence], "source": {"kind": "sequence"}}

        with mock.patch.dict(Residue_Data.PROVIDERS, {"hydrophobic": hydrophobic}):
            result = Residue_Data.compute_residue_data(inputs_for(HAND_ROWS), 0, "ACDEFGH")
        self.assertEqual(list(result.properties), ["information_content", "same_residue_pct", "hydrophobic"])
        self.assertEqual(result.properties["hydrophobic"]["values"], [1, 0, 0, 0, 0, 0, 0])
        self.assertNotIn("hydrophobic", Residue_Data.PROVIDERS)

    def test_a_provider_with_nothing_to_say_is_left_out_without_a_reason(self):
        with mock.patch.dict(Residue_Data.PROVIDERS, {"nothing": lambda *args: None}, clear=True):
            result = Residue_Data.compute_residue_data(inputs_for(HAND_ROWS), 0, "ACDEFGH")
        self.assertEqual((result.properties, result.unavailable), ({}, {}))

    def test_a_provider_that_fails_costs_only_its_own_property(self):
        def broken(inputs, node_index, sequence):
            raise RuntimeError("boom")

        with mock.patch.dict(Residue_Data.PROVIDERS, {"broken": broken}):
            result = Residue_Data.compute_residue_data(inputs_for(HAND_ROWS), 0, "ACDEFGH")
        self.assertEqual(list(result.properties), ["information_content", "same_residue_pct"])
        self.assertEqual(result.unavailable["broken"].kind, Residue_Data.FAILED)
        self.assertEqual(result.unavailable["broken"].reason, "RuntimeError: boom")


class AlignmentCostTests(unittest.TestCase):
    """What is worked out about an alignment is worked out once, however many structures are folded."""

    def test_the_columns_are_counted_once_per_alignment(self):
        inputs = inputs_for(HAND_ROWS)
        with mock.patch.object(Residue_Data, "_column_counts", wraps=Residue_Data._column_counts) as counted:
            for node in range(4):
                Residue_Data.compute_residue_data(inputs, node, ungapped(HAND_ROWS[node][1]))
        self.assertEqual(counted.call_count, 1)

        # Another alignment object is another alignment, even with the same rows.
        again = inputs_for(HAND_ROWS)
        with mock.patch.object(Residue_Data, "_column_counts", wraps=Residue_Data._column_counts) as counted:
            Residue_Data.compute_residue_data(again, 0, "ACDEFGH")
        self.assertEqual(counted.call_count, 1)

    def test_a_matrix_replaced_in_the_same_alignment_is_counted_again(self):
        inputs = inputs_for(HAND_ROWS)
        self.assertEqual(Residue_Data.same_residue_pct(inputs, 0, "ACDEFGH")["values"][0], 75.0)  # A A A G
        aln = inputs.alignment.aln
        dense = aln.matrix.toarray()
        dense[3, 0] = INT_TO_AA_CODE["A"]  # n3's G in column 0 becomes an A.
        aln.matrix = sparse.csr_matrix(dense)
        self.assertEqual(Residue_Data.same_residue_pct(inputs, 0, "ACDEFGH")["values"][0], 100.0)

    def test_an_alignment_of_many_rows_costs_one_pass_and_no_work_per_row_and_column(self):
        rows, columns = 6000, 800
        rng = np.random.default_rng(42)
        codes = rng.integers(1, 21, size=(rows, columns), dtype=np.uint8)
        codes[rng.random((rows, columns)) < 0.3] = 0
        aln = SimpleNamespace(
            matrix=sparse.csr_matrix(codes), int_to_aa=dict(INT_TO_AA), headers=[f"s{i}" for i in range(rows)]
        )
        inputs = Residue_Data.ViewerInputs(SimpleNamespace(aln=aln, msa_file="big.fasta"), np.arange(rows), ())

        def fold(node):
            sequence = "".join(INT_TO_AA[int(code)] for code in codes[node] if code)
            begin = time.perf_counter()
            result = Residue_Data.compute_residue_data(inputs, node, sequence)
            return time.perf_counter() - begin, result

        first, result = fold(10)
        later = [fold(node)[0] for node in (11, 12, 13)]
        self.assertEqual(result.unavailable, {})
        # Bounds are loose on purpose: a Python loop over rows x columns would take minutes.
        self.assertLess(first, 10.0)
        self.assertLess(max(later), 1.0)


if __name__ == "__main__":
    unittest.main()
