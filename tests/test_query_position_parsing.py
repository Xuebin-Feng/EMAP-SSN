"""Focused tests for query position-list syntax."""

import os
import re
import sys
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands.query import parse_query_positions


VALID_LABELS = [
    ((-3, 0), "-3"),
    ((-2, 0), "-2"),
    ((-1, 0), "-1"),
    ((-1, 1), "-1.1"),
    ((0, 0), "0"),
    ((1, 0), "1"),
    ((1, 1), "1.1"),
    ((2, 0), "2"),
]


class QueryPositionParsingTests(unittest.TestCase):
    def test_parenthesized_negative_positions_are_normalized(self):
        self.assertEqual(
            parse_query_positions("[(-1),(-1.1),0]", VALID_LABELS),
            ["-1", "-1.1", "0"],
        )

    def test_negative_ranges_expand_over_mapped_insertion_labels(self):
        self.assertEqual(
            parse_query_positions("[(-3)-(-1)]", VALID_LABELS),
            ["-3", "-2", "-1"],
        )
        self.assertEqual(
            parse_query_positions("[(-2)-1]", VALID_LABELS),
            ["-2", "-1", "-1.1", "0", "1"],
        )

    def test_descending_ranges_and_end_alias_remain_supported(self):
        self.assertEqual(
            parse_query_positions("[(-1)-(-3)]", VALID_LABELS),
            ["-3", "-2", "-1"],
        )
        self.assertEqual(
            parse_query_positions("[END-1]", VALID_LABELS),
            ["1", "1.1", "2"],
        )
        self.assertEqual(
            parse_query_positions("[E,0]", VALID_LABELS),
            ["2", "0"],
        )

    def test_bare_negative_positions_are_rejected_with_correction(self):
        for position_spec, correction in (
            ("[-1]", "(-1)"),
            ("[-1.1,0]", "(-1.1)"),
            ("[-3--1]", "(-3)"),
            ("[1--1]", "(-1)"),
        ):
            with self.subTest(position_spec=position_spec):
                with self.assertRaisesRegex(ValueError, re.escape(correction)):
                    parse_query_positions(position_spec, VALID_LABELS)


if __name__ == "__main__":
    unittest.main()
