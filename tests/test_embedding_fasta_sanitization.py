import contextlib
import io
import pathlib
import tempfile
import unittest


from src.utilities.FASTA_Sanitization import (
    load_sanitized_fasta,
    print_sanitization_result,
    sanitize_fasta_records,
)


class EmbeddingFastaSanitizationTests(unittest.TestCase):
    def test_clean_records_do_not_print_a_sanitization_result(self):
        headers, sequences, stats = sanitize_fasta_records(
            ["Clean_header", "Another header"],
            ["ACDE", "FGHI"],
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            printed = print_sanitization_result(stats)

        self.assertEqual(headers, ["Clean_header", "Another header"])
        self.assertEqual(sequences, ["ACDE", "FGHI"])
        self.assertFalse(printed)
        self.assertEqual(output.getvalue(), "")

    def test_modified_records_print_a_compact_result(self):
        headers, sequences, stats = sanitize_fasta_records(
            ["Alpha??__Beta"],
            ["ac-d"],
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            printed = print_sanitization_result(stats)

        self.assertEqual(headers, ["Alpha_Beta"])
        self.assertEqual(sequences, ["ACXD"])
        self.assertTrue(printed)
        self.assertIn("FASTA sanitization result:", output.getvalue())
        self.assertIn("Headers modified:", output.getvalue())
        self.assertIn("Sequences modified:", output.getvalue())

    def test_embedding_sanitization_has_no_header_or_length_filter(self):
        short_sequence = "A"
        long_sequence = "C" * 1000

        headers, sequences, stats = sanitize_fasta_records(
            ["remove_me", "keep_me"],
            [short_sequence, long_sequence],
        )

        self.assertEqual(headers, ["remove_me", "keep_me"])
        self.assertEqual(sequences, [short_sequence, long_sequence])
        self.assertFalse(stats["changed"])

    def test_deduplication_keeps_headers_unique(self):
        headers, sequences, stats = sanitize_fasta_records(
            ["A", "A", "A_1"],
            ["AAA", "CCC", "GGG"],
        )

        self.assertEqual(headers, ["A_2", "A_3", "A_1"])
        self.assertEqual(sequences, ["AAA", "CCC", "GGG"])
        self.assertEqual(len(headers), len(set(headers)))
        self.assertEqual(stats["headers_renamed"], 2)

    def test_generated_suffix_does_not_reintroduce_repeated_underscores(self):
        headers, _, _ = sanitize_fasta_records(
            ["A?", "A#"],
            ["AAA", "CCC"],
        )

        self.assertEqual(headers, ["A_1", "A_2"])
        self.assertTrue(all("__" not in header for header in headers))

    def test_file_loader_is_silent_for_clean_fasta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = pathlib.Path(temp_dir) / "clean.fasta"
            fasta_path.write_text(">Alpha\nACDE\n", encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                headers, sequences, stats = load_sanitized_fasta(fasta_path)

        self.assertEqual(headers, ["Alpha"])
        self.assertEqual(sequences, ["ACDE"])
        self.assertFalse(stats["changed"])
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
