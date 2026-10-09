"""Tests for the benchmark that ships in src/resources/benchmark: its sequence sets."""

import hashlib
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities.Sequence_Utils import VALID_RESIDUE_CODES, read_fasta, sanitize_fasta_records

BENCHMARK_DIR = SRC_DIR / "resources" / "benchmark"
README = BENCHMARK_DIR / "README.md"
MAIN_SET = BENCHMARK_DIR / "benchmark_sequences.fasta"
INJECTION_SET = BENCHMARK_DIR / "injection_sequences.fasta"
ESM2_LONGEST_SEQUENCE = 1022
GLYOXALASE_II = " Hydroxyacylglutathione hydrolase OS="


def recorded_files():
    """{file name: (bytes, SHA-256)} from the README's table of the bundled files."""
    text = README.read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([^`]+\.fasta)` \| ([\d,]+) \| `([0-9a-f]{64})` \|\s*$", text, re.MULTILINE)
    return {name: (int(size.replace(",", "")), digest) for name, size, digest in rows}


def readme_recipe():
    """The README's Python code that makes both files from the UniProt download."""
    [code] = re.findall(r"^```python\n(.*?)^```", README.read_text(encoding="utf-8"), re.MULTILINE | re.DOTALL)
    return code


def accession(header):
    return header.split("|")[1]


class BenchmarkSequenceSetTests(unittest.TestCase):
    """The two FASTA files are the ones the README describes, byte for byte."""

    @classmethod
    def setUpClass(cls):
        cls.main_headers, cls.main_sequences = read_fasta(MAIN_SET)
        cls.injection_headers, cls.injection_sequences = read_fasta(INJECTION_SET)

    def test_the_files_match_the_sizes_and_checksums_the_readme_records(self):
        recorded = recorded_files()
        self.assertEqual(set(recorded), {MAIN_SET.name, INJECTION_SET.name})
        for path in (MAIN_SET, INJECTION_SET):
            with self.subTest(path=path.name):
                data = path.read_bytes()
                self.assertEqual((len(data), hashlib.sha256(data).hexdigest()), recorded[path.name])

    def test_the_sets_hold_860_and_92_records_with_unique_headers(self):
        self.assertEqual(len(self.main_headers), 860)
        self.assertEqual(len(self.injection_headers), 92)
        headers = self.main_headers + self.injection_headers
        self.assertEqual(len(set(headers)), len(headers))
        self.assertEqual(len({accession(header) for header in headers}), len(headers))

    def test_every_sequence_is_distinct_valid_and_short_enough_for_esm_2(self):
        sequences = self.main_sequences + self.injection_sequences
        self.assertEqual(len(set(sequences)), len(sequences), "no sequence appears twice in or across the sets")
        for header, sequence in zip(self.main_headers + self.injection_headers, sequences):
            with self.subTest(header=header):
                self.assertTrue(sequence)
                self.assertLessEqual(len(sequence), ESM2_LONGEST_SEQUENCE)
                self.assertEqual(set(sequence) - set(VALID_RESIDUE_CODES), set())

    def test_sanitizing_keeps_every_record_and_changes_only_the_headers(self):
        for headers, sequences in (
            (self.main_headers, self.main_sequences),
            (self.injection_headers, self.injection_sequences),
        ):
            clean_headers, clean_sequences, stats = sanitize_fasta_records(headers, sequences)
            self.assertEqual(clean_sequences, sequences)
            self.assertEqual(len(set(clean_headers)), len(headers))
            self.assertEqual(stats["final_records"], len(headers))
            self.assertEqual(stats["headers_modified"], len(headers), "UniProt headers hold spaces")
            for key in ("sequences_modified", "empty_sequences_removed", "exact_duplicates_removed",
                        "different_headers_merged", "headers_renamed"):
                self.assertEqual(stats[key], 0, key)

    def test_the_records_keep_uniprot_s_headers_and_60_residue_lines(self):
        for path in (MAIN_SET, INJECTION_SET):
            with self.subTest(path=path.name):
                text = path.read_bytes().decode("ascii")
                self.assertNotIn("\r", text)
                self.assertTrue(text.endswith("\n"))
                for record in re.findall(r">[^>]*", text):
                    header, *lines = record[:-1].split("\n")
                    self.assertRegex(header, r"^>sp\|[A-Z0-9]+\|\S+ .+ OS=.+ OX=\d+( GN=\S+)? PE=\d SV=\d+$")
                    self.assertTrue(all(len(line) == 60 for line in lines[:-1]))
                    self.assertTrue(0 < len(lines[-1]) <= 60)

    def test_the_main_set_holds_the_glyoxalase_ii_entries_the_msa_stage_aligns(self):
        entries = sorted(accession(header) for header in self.main_headers if GLYOXALASE_II in header)
        self.assertEqual(len(entries), 196)
        self.assertEqual((entries[0], entries[99]), ("A0KIK2", "Q04RQ6"), "the README's first 100")

    def test_the_readme_recipe_keeps_the_first_of_each_sequence_and_drops_long_ones(self):
        record = ">sp|{}|T_X Protein OS=Organism OX=1 PE=1 SV=1\n{}\n".format
        long_sequence = "M" * (ESM2_LONGEST_SEQUENCE + 1)
        download = "".join((
            record("Q00003", "MKV"),
            record("Q00001", "MKV"),
            record("Q00002", "MKA"),
            record("Q00004", "\n".join(long_sequence[i:i + 60] for i in range(0, len(long_sequence), 60))),
            record("Q00005", "MKV"),
        ))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "IPR001279_reviewed.fasta").write_bytes(download.encode("ascii"))
            (folder / "recipe.py").write_text(readme_recipe(), encoding="utf-8")
            subprocess.run([sys.executable, "recipe.py"], cwd=folder, check=True, capture_output=True)
            headers, sequences = read_fasta(folder / MAIN_SET.name)
            self.assertEqual(sorted(zip(map(accession, headers), sequences)),
                             [("Q00001", "MKV"), ("Q00002", "MKA")])
            self.assertEqual((folder / INJECTION_SET.name).read_bytes(), b"")

    def test_the_readme_recipe_makes_both_files_from_their_records(self):
        # The sets hold no copies and nothing over 1,022 residues, so the recipe
        # keeps all 952 records, sorts them by accession and must shuffle them
        # back into the bundled order and split.
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "IPR001279_reviewed.fasta").write_bytes(INJECTION_SET.read_bytes() + MAIN_SET.read_bytes())
            (folder / "recipe.py").write_text(readme_recipe(), encoding="utf-8")
            subprocess.run([sys.executable, "recipe.py"], cwd=folder, check=True, capture_output=True)
            for path in (MAIN_SET, INJECTION_SET):
                with self.subTest(path=path.name):
                    self.assertEqual((folder / path.name).read_bytes(), path.read_bytes())


class BenchmarkRepositoryTests(unittest.TestCase):
    """Git keeps the sets' bytes and ignores what a run leaves in the folder."""

    def git(self, *arguments):
        try:
            result = subprocess.run(["git", "-C", str(PROJECT_ROOT), *arguments], capture_output=True, text=True)
        except OSError:
            self.skipTest("git is not installed")
        if result.returncode not in (0, 1):
            self.skipTest("not a git checkout")
        return result

    def test_temporary_files_and_reports_are_ignored_and_the_bundled_files_are_not(self):
        folder = "src/resources/benchmark/"
        for name in ("temp/benchmark.lock", "temp/logs/stage_03.log", "Benchmark_Report_2026-10-09_15-00-00.txt",
                     "Benchmark_Report_2026-10-09_15-00-00.json", "Benchmark_Report_2026-10-09_15-00-00_2.txt"):
            with self.subTest(name=name):
                self.assertEqual(self.git("check-ignore", "--no-index", "--quiet", "--", folder + name).returncode, 0)
        for name in ("benchmark_sequences.fasta", "injection_sequences.fasta", "README.md"):
            with self.subTest(name=name):
                self.assertEqual(self.git("check-ignore", "--no-index", "--quiet", "--", folder + name).returncode, 1)

    def test_git_never_converts_the_sets_line_endings(self):
        for path in (MAIN_SET, INJECTION_SET):
            with self.subTest(path=path.name):
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                self.assertEqual(self.git("check-attr", "text", "--", relative).stdout.strip(),
                                 f"{relative}: text: unset")


if __name__ == "__main__":
    unittest.main()
