"""Position lists in utilities.Sequence_Utils, which `logo` and `query` share.

split_position_list reads a bracketed list of positions, comma-separated or
space-separated, and rejoin_split_brackets puts back a bracket that the
command line split at its spaces. What each command does with the items is in
test_logo_command.py and test_query_command.py.
"""

import os
import re
import sys
import unittest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utilities.Localization import Message
from utilities.Sequence_Utils import rejoin_split_brackets, split_position_list


# Each row is one list written with commas, the same list written with spaces,
# and the items both give.
BOTH_FORMS = (
    ("[1,5,9-12]", "[1 5 9-12]", ["1", "5", "9-12"]),
    ("[1, 5, 9-12]", "[1   5   9-12]", ["1", "5", "9-12"]),
    ("[10,10.1,10.2,11]", "[10 10.1 10.2 11]", ["10", "10.1", "10.2", "11"]),
    ("[(-3)-(-1),0,5]", "[(-3)-(-1) 0 5]", ["(-3)-(-1)", "0", "5"]),
    ("[(-1),(-1.1),0]", "[(-1) (-1.1) 0]", ["(-1)", "(-1.1)", "0"]),
    ("[(-3)-2,5]", "[(-3)-2 5]", ["(-3)-2", "5"]),
    ("[+1,+1.1]", "[+1 +1.1]", ["+1", "+1.1"]),
    ("[250-E,END]", "[250-E END]", ["250-E", "END"]),
    ("[E,0,END-1]", "[E 0 END-1]", ["E", "0", "END-1"]),
    ("[7]", "[7]", ["7"]),
)


class SpaceAndCommaListTests(unittest.TestCase):
    def test_a_list_gives_the_same_items_with_commas_or_with_spaces(self):
        for commas, spaces, items in BOTH_FORMS:
            with self.subTest(commas=commas):
                self.assertEqual(split_position_list(commas), items)
            with self.subTest(spaces=spaces):
                self.assertEqual(split_position_list(spaces), items)

    def test_spaces_just_inside_the_brackets_are_ignored_in_both_forms(self):
        for text in (
            "[ 1, 5 ]", "[1, 5 ]", "[ 1,5]", "[   1,   5   ]", "[1 ,5]", "[1 , 5]",
            "[ 1 5 ]", "[1 5 ]", "[ 1 5]", "[   1   5   ]", " [1 5] ", "[\t1\t5\t]",
        ):
            with self.subTest(text=text):
                self.assertEqual(split_position_list(text), ["1", "5"])

    def test_the_brackets_may_be_left_off(self):
        self.assertEqual(split_position_list("1,5,9-12"), ["1", "5", "9-12"])
        self.assertEqual(split_position_list("1 5 9-12"), ["1", "5", "9-12"])
        self.assertEqual(split_position_list("  1 5  "), ["1", "5"])

    def test_an_empty_list_gives_no_items_as_it_always_did(self):
        for text in ("[]", "[ ]", "[,]", "[ , ]", "", "   "):
            with self.subTest(text=text):
                self.assertEqual(split_position_list(text), [])

    def test_empty_items_between_commas_are_skipped(self):
        self.assertEqual(split_position_list("[1,,5]"), ["1", "5"])
        self.assertEqual(split_position_list("[,1,5,]"), ["1", "5"])
        self.assertEqual(split_position_list("[1 , , 5]"), ["1", "5"])


class RangeDashTests(unittest.TestCase):
    def test_spaces_around_a_ranges_dash_belong_to_the_range(self):
        for text, items in (
            ("[9 - 12]", ["9-12"]),
            ("[9- 12]", ["9-12"]),
            ("[9 -(-3)]", ["9-(-3)"]),
            ("[(-3) - 2]", ["(-3)-2"]),
            ("[(-3) - (-1)]", ["(-3)-(-1)"]),
            ("[250 - E]", ["250-E"]),
            ("[END - 1]", ["END-1"]),
            ("[ 9 - 12 ]", ["9-12"]),
            # In a comma-separated list.
            ("[1, 9 - 12]", ["1", "9-12"]),
            ("[1, 9 - 12, 20]", ["1", "9-12", "20"]),
            ("[1 , 9- 12 , 20 - 25]", ["1", "9-12", "20-25"]),
            # In a space-separated list.
            ("[1 9 - 12]", ["1", "9-12"]),
            ("[1 9- 12 20]", ["1", "9-12", "20"]),
            ("[1 9 - 12 20 - 25 30]", ["1", "9-12", "20-25", "30"]),
            ("[9 - 12 1]", ["9-12", "1"]),
            ("[250 - E END]", ["250-E", "END"]),
            ("[(-3) - (-1) 0 5]", ["(-3)-(-1)", "0", "5"]),
        ):
            with self.subTest(text=text):
                self.assertEqual(split_position_list(text), items)

    def test_the_dash_of_a_parenthesised_negative_is_not_a_range_dash(self):
        self.assertEqual(split_position_list("[(-3) (-1)]"), ["(-3)", "(-1)"])
        self.assertEqual(split_position_list("[(-3), (-1)]"), ["(-3)", "(-1)"])
        self.assertEqual(split_position_list("[1 (-1) 3]"), ["1", "(-1)", "3"])
        # A space inside the parentheses was never allowed, and still is not.
        self.assertEqual(split_position_list("[(- 3)]"), ["(-", "3)"])

    def test_a_dash_with_a_space_on_its_left_only_is_a_bare_negative(self):
        for text, negative in (
            ("[1 -5]", "-5"),
            ("[1 5 -3]", "-3"),
            ("[9 -12]", "-12"),
            ("[1, -5]", "-5"),
            ("[1 -5.1]", "-5.1"),
            ("[-1]", "-1"),
            ("[-1 2]", "-1"),
            ("[1 2 -3 4]", "-3"),
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as caught:
                    split_position_list(text)
                self.assertIsInstance(caught.exception.args[0], Message)
                self.assertIn(f"'({negative})'", str(caught.exception))

    def test_a_dash_without_spaces_is_a_range_not_a_negative(self):
        self.assertEqual(split_position_list("[1-5 7]"), ["1-5", "7"])
        self.assertEqual(split_position_list("[1 -(-5)]"), ["1-(-5)"])


class MixedSeparatorTests(unittest.TestCase):
    def test_a_comma_list_with_a_space_inside_an_item_is_refused_naming_the_item(self):
        for text, item in (
            ("[1, 5 9]", "5 9"),
            ("[1 5, 9]", "1 5"),
            ("[1 5,]", "1 5"),
            ("[,1 5]", "1 5"),
            ("[1,5 9,12]", "5 9"),
            ("[1, 5, 9 12]", "9 12"),
            ("[1,5 ,9 12]", "9 12"),
            ("[1 2 3, 4]", "1 2 3"),
            ("[10.1 10.2, 11]", "10.1 10.2"),
            ("[1, 9 - 12 20]", "9-12 20"),
            ("[1, (-3) 0]", "(-3) 0"),
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as caught:
                    split_position_list(text)
                error = caught.exception
                self.assertIsInstance(error.args[0], Message)
                self.assertIn(f"'{item}'", str(error))
                self.assertIn("commas or with spaces, not both", str(error))

    def test_the_refusal_is_in_the_catalog_and_fills_in_the_item(self):
        with self.assertRaises(ValueError) as caught:
            split_position_list("[1, 5 9]")
        message = caught.exception.args[0]

        self.assertEqual(message.values, {"item": "5 9"})
        self.assertIn("{item}", message.template)
        self.assertEqual(
            str(message),
            "Separate positions with commas or with spaces, not both: '5 9' has a "
            "space inside a comma-separated list. Write [1,5,9-12] or [1 5 9-12].",
        )

    def test_nothing_that_is_not_mixed_is_refused(self):
        for text in (
            "[1,5,9-12]", "[1 5 9-12]", "[ 1 , 5 ]", "[1 9 - 12]", "[1, 9 - 12]",
            "[1 5 9 - 12 ]", "[ , ]",
        ):
            with self.subTest(text=text):
                split_position_list(text)


class RejoinSplitBracketsTests(unittest.TestCase):
    def test_tokens_of_one_bracket_are_joined_with_single_spaces(self):
        for tokens, joined in (
            (["[1,", "5]"], ["[1, 5]"]),
            (["[1", "5", "9-12]"], ["[1 5 9-12]"]),
            (["[", "1", "5", "]"], ["[ 1 5 ]"]),
            (["[", "1,5", "]"], ["[ 1,5 ]"]),
            (["[1", ",", "3", "]"], ["[1 , 3 ]"]),
            (["[1", "9", "-", "12]"], ["[1 9 - 12]"]),
        ):
            with self.subTest(tokens=tokens):
                self.assertEqual(rejoin_split_brackets(tokens), joined)

    def test_other_arguments_stay_where_they_were(self):
        self.assertEqual(
            rejoin_split_brackets(["#cluster_1#", "[1", "5]", "pcts", "out.svg"]),
            ["#cluster_1#", "[1 5]", "pcts", "out.svg"],
        )
        self.assertEqual(
            rejoin_split_brackets(["5", "[1", "5]", "90%"]),
            ["5", "[1 5]", "90%"],
        )

    def test_a_bracket_that_is_whole_is_left_alone(self):
        self.assertEqual(rejoin_split_brackets(["[1,5]", "5"]), ["[1,5]", "5"])
        self.assertEqual(rejoin_split_brackets([]), [])

    def test_brackets_that_never_balance_come_back_as_they_were(self):
        self.assertEqual(rejoin_split_brackets(["[1", "5", "9"]), ["[1", "5", "9"])
        self.assertEqual(
            rejoin_split_brackets(["[1-2]", "[3,", "out.svg"]),
            ["[1-2]", "[3,", "out.svg"],
        )

    def test_a_logic_bracket_is_joined_whole(self):
        self.assertEqual(
            rejoin_split_brackets(["[(K>0.1)", "&", "(R>0.05)]"]),
            ["[(K>0.1) & (R>0.05)]"],
        )


class ExactTextTests(unittest.TestCase):
    """The text the error holds, as the commands' windows show it."""

    def test_the_bare_negative_message_is_the_one_it_always_was(self):
        with self.assertRaises(ValueError) as caught:
            split_position_list("[1 -5]")

        self.assertEqual(
            str(caught.exception),
            "Negative position '-5' must be written as '(-5)'. "
            "Parentheses are required around negative positions.",
        )
        self.assertTrue(re.search(r"'\(-5\)'", str(caught.exception)))


if __name__ == "__main__":
    unittest.main()
