"""Tests for the benchmark that ships in src/resources/benchmark: its sequence sets and Run_Benchmark.py."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import importlib.util
from io import StringIO
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities import Localization
from utilities.Sequence_Utils import VALID_RESIDUE_CODES, read_fasta, sanitize_fasta_records

BENCHMARK_DIR = SRC_DIR / "resources" / "benchmark"
README = BENCHMARK_DIR / "README.md"
MAIN_SET = BENCHMARK_DIR / "benchmark_sequences.fasta"
INJECTION_SET = BENCHMARK_DIR / "injection_sequences.fasta"
ESM2_LONGEST_SEQUENCE = 1022
GLYOXALASE_II = " Hydroxyacylglutathione hydrolase OS="


def load_script():
    """Run_Benchmark.py as a module; src/resources is no package, so it loads from its path."""
    spec = importlib.util.spec_from_file_location("Run_Benchmark", BENCHMARK_DIR / "Run_Benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # Its dataclasses look their module up by name.
    spec.loader.exec_module(module)
    return module


benchmark = load_script()


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
        for name in ("benchmark_sequences.fasta", "injection_sequences.fasta", "README.md", "Run_Benchmark.py"):
            with self.subTest(name=name):
                self.assertEqual(self.git("check-ignore", "--no-index", "--quiet", "--", folder + name).returncode, 1)

    def test_git_never_converts_the_sets_line_endings(self):
        for path in (MAIN_SET, INJECTION_SET):
            with self.subTest(path=path.name):
                relative = path.relative_to(PROJECT_ROOT).as_posix()
                self.assertEqual(self.git("check-attr", "text", "--", relative).stdout.strip(),
                                 f"{relative}: text: unset")


class BenchmarkFolderTests(unittest.TestCase):
    """temp/ holds one run's files under a lock and is cleared; reports are never overwritten."""

    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.temp = self.folder / "temp"

    def test_the_tests_never_use_the_real_benchmark_folder(self):
        self.assertNotEqual(benchmark.work_dir(), benchmark.BENCHMARK_DIR)

    def test_the_work_folder_follows_its_variable_and_is_otherwise_the_script_s_folder(self):
        with mock.patch.dict(os.environ, {benchmark.WORK_DIR_VARIABLE: str(self.folder)}):
            self.assertEqual(benchmark.work_dir(), self.folder.resolve())
        with mock.patch.dict(os.environ):
            os.environ.pop(benchmark.WORK_DIR_VARIABLE, None)
            self.assertEqual(benchmark.work_dir(), BENCHMARK_DIR)

    def test_a_run_clears_what_an_earlier_run_left_and_removes_temp_at_its_end(self):
        (self.temp / "embeddings").mkdir(parents=True)
        (self.temp / "embeddings" / "left_behind.h5").write_bytes(b"old")
        (self.temp / "context.json").write_text("{}", encoding="utf-8")
        lock = benchmark.acquire_lock(self.temp)
        benchmark.wipe_temp(self.temp, keep=(benchmark.LOCK_NAME,))
        self.assertEqual([path.name for path in self.temp.iterdir()], [benchmark.LOCK_NAME])
        (self.temp / "logs").mkdir()
        (self.temp / "logs" / "stage_01_sanitize.log").write_text("output", encoding="utf-8")
        benchmark.remove_temp(self.temp, lock)
        self.assertFalse(self.temp.exists())

    def test_a_live_lock_stops_a_second_run(self):
        lock = benchmark.acquire_lock(self.temp)
        self.addCleanup(benchmark.release_lock, lock)
        with self.assertRaises(benchmark.CannotStart):
            benchmark.acquire_lock(self.temp)
        self.assertEqual(benchmark.read_lock(lock)["pid"], os.getpid())

    def test_a_lock_whose_process_is_gone_is_taken_over(self):
        self.temp.mkdir()
        path = self.temp / benchmark.LOCK_NAME
        # The same process ID with another start time: the ID now belongs to a new process.
        for stale in (json.dumps({"pid": os.getpid(), "started": 1.0, "host": socket.gethostname()}), "{cut short"):
            with self.subTest(lock=stale):
                path.write_text(stale, encoding="utf-8")
                lock = benchmark.acquire_lock(self.temp)
                self.assertEqual(benchmark.read_lock(lock)["pid"], os.getpid())
                benchmark.release_lock(lock)
                self.assertFalse(lock.exists())

    @unittest.skipUnless(os.name == "nt", "Windows refuses to delete a file a program has open")
    def test_a_file_still_in_use_stops_the_start_instead_of_running_on_old_files(self):
        self.temp.mkdir()
        with open(self.temp / "busy.h5", "wb"):
            with self.assertRaises(benchmark.CannotStart) as raised:
                benchmark.wipe_temp(self.temp)
        self.assertIn("busy.h5", str(raised.exception.args[0]))

    def test_too_little_free_space_stops_the_start(self):
        with self.assertRaises(benchmark.CannotStart):
            benchmark.check_disk(self.folder, required=1 << 62)

    def test_a_report_never_takes_the_name_of_an_earlier_one(self):
        started = datetime(2026, 10, 9, 21, 0, 0)
        first = benchmark.reserve_report_paths(self.folder, started)
        second = benchmark.reserve_report_paths(self.folder, started)
        self.assertEqual([path.name for path in first],
                         ["Benchmark_Report_2026-10-09_21-00-00.txt", "Benchmark_Report_2026-10-09_21-00-00.json"])
        self.assertEqual([path.name for path in second],
                         ["Benchmark_Report_2026-10-09_21-00-00_2.txt", "Benchmark_Report_2026-10-09_21-00-00_2.json"])
        (self.folder / "Benchmark_Report_2026-10-09_21-00-00_3.json").write_text("{}", encoding="utf-8")
        third = benchmark.reserve_report_paths(self.folder, started)
        self.assertEqual(third[0].name, "Benchmark_Report_2026-10-09_21-00-00_4.txt")


class BenchmarkInputTests(unittest.TestCase):
    """The sizes of the work, the search query and the MSA subset."""

    def test_an_all_against_all_run_counts_pairs_and_dynamic_programming_cells(self):
        self.assertEqual(
            benchmark.all_pairs_workload(["AA", "AAA", "A"]),
            {"sequences": 3, "residues": 6, "pairs": 3, "cells": 2 * 3 + 2 * 1 + 3 * 1},
        )

    def test_injection_aligns_the_new_sequences_with_the_old_ones_and_each_other(self):
        self.assertEqual(
            benchmark.injection_workload(["AA", "AAA"], ["A", "AAAA"]),
            {"sequences": 2, "residues": 5, "pairs": 2 * 2 + 1, "cells": 5 * 5 + 1 * 4},
        )

    def test_the_full_run_searches_with_e_coli_glyoxalase_ii_and_aligns_100_of_its_kind(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs = benchmark.prepare_inputs(Path(folder))
            self.assertEqual(list(Path(folder).iterdir()), [], "the bundled files are read where they are")
        self.assertEqual(inputs["main_fasta"], str(MAIN_SET))
        self.assertEqual(accession(inputs["search_query"]), "B1XD76")
        self.assertEqual(len(inputs["msa_headers"]), 100)
        self.assertTrue(all(GLYOXALASE_II in header for header in inputs["msa_headers"]))
        self.assertEqual(inputs["workload"]["main"]["pairs"], 860 * 859 // 2)
        self.assertEqual(inputs["workload"]["injection"]["pairs"], 92 * 860 + 92 * 91 // 2)
        self.assertIsNone(inputs["dataset"]["limit"])

    def test_the_test_option_takes_the_first_records_into_temp(self):
        main_headers, _ = read_fasta(MAIN_SET)
        with tempfile.TemporaryDirectory() as folder:
            inputs = benchmark.prepare_inputs(Path(folder), limit=20)
            headers, _ = read_fasta(inputs["main_fasta"])
            new_headers, _ = read_fasta(inputs["injection_fasta"])
        self.assertEqual(headers, main_headers[:20])
        self.assertEqual(len(new_headers), 2)
        self.assertEqual(inputs["search_query"], main_headers[0], "B1XD76 is not among the first 20")
        self.assertEqual(inputs["dataset"]["limit"], 20)


class BenchmarkStageTableTests(unittest.TestCase):
    """The ten stages, what each needs, and which a run takes."""

    def test_the_ten_stages_run_in_order_and_each_needs_only_earlier_ones(self):
        self.assertEqual([stage.number for stage in benchmark.STAGES], list(range(1, 11)))
        seen = set()
        for stage in benchmark.STAGES:
            with self.subTest(stage=stage.key):
                self.assertLessEqual(set(stage.needs), seen)
                self.assertTrue(stage.steps)
                self.assertLessEqual(set(stage.steps), set(benchmark.STEP_RUNNERS))
                self.assertLessEqual(set(stage.auto), set(benchmark.DECISION_TITLES))
            seen.add(stage.key)

    def test_choosing_stages_adds_the_stages_they_need(self):
        everything = [stage.key for stage in benchmark.STAGES]
        self.assertEqual(benchmark.select_stages(None), everything)
        self.assertEqual(benchmark.select_stages("4"), ["sanitize", "embeddings", "alignment", "ssn_layout"])
        self.assertEqual(benchmark.select_stages(" blast, 1 "), ["sanitize", "blast"])
        self.assertEqual(benchmark.select_stages("injection"), ["sanitize", "embeddings", "alignment", "injection"])

    def test_an_unknown_stage_stops_the_start(self):
        for text in ("11", "nope"):
            with self.subTest(text=text), self.assertRaises(benchmark.CannotStart):
                benchmark.select_stages(text)


class BenchmarkOutputParsingTests(unittest.TestCase):
    """What the report reads from the tools' own output."""

    def log(self, text):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (folder / "stage.log").write_bytes(text.encode("utf-8"))
        return folder / "stage.log"

    def test_layout_stop_lines(self):
        path = self.log(
            "Simulating Batch 1/2 (4 components, 11 nodes)...\r\n"
            "    - Step 0500/10000: RMSD = 0.03639\n"
            "    - Converged at Step 1291 (RMSD: 0.00499)\n"
            "    - Plateau Reached at Step 99 (RMSD: 0.1)\n"
            "    - Step limit reached after 10000 steps (RMSD: 0.01044)\n"
        )
        self.assertEqual(benchmark.layout_stops(path), [
            {"stop": "Converged", "steps": 1292},
            {"stop": "Plateau Reached", "steps": 100},
            {"stop": "Step limit reached", "steps": 10000},
        ])

    def test_msa_time_lines(self):
        path = self.log("Total processing time: 2m 5.00s\nTree building time: 1h 1m 2.50s\nCluster merging time: 0.85s\n")
        times = benchmark.msa_times(path)
        self.assertAlmostEqual(times["tree_seconds"], 3662.5)
        self.assertAlmostEqual(times["merge_seconds"], 0.85)


class BenchmarkStageProcessTests(unittest.TestCase):
    """A step runs in a process of its own, measured from outside and reported from inside."""

    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_a_stage_process_is_logged_and_measured(self):
        run = benchmark.run_stage_process(
            [sys.executable, "-c", "print('hello from a stage'); raise SystemExit(3)"],
            cwd=self.folder, env=dict(os.environ), log_path=self.folder / "logs" / "stage.log", echo=False,
        )
        self.assertEqual(run.returncode, 3)
        self.assertFalse(run.interrupted)
        self.assertGreater(run.wall_seconds, 0)
        self.assertGreater(run.peak_rss_bytes, 0)
        self.assertGreaterEqual(run.peak_processes, 1)
        self.assertEqual(benchmark.log_tail(run.log), ["hello from a stage"])

    def test_a_step_writes_its_result_even_when_it_fails(self):
        context_path = self.folder / "context.json"
        context_path.write_text(json.dumps({"temp": str(self.folder), "stage": 1}), encoding="utf-8")

        def broken(_context):
            raise RuntimeError("broken")

        runners = {"good": lambda _context: {"returncode": 0, "tool_seconds": 1.5, "outputs": {"x": "y"}},
                   "broken": broken}
        with mock.patch.dict(benchmark.STEP_RUNNERS, runners), redirect_stderr(StringIO()):
            self.assertEqual(benchmark.stage_main("good", context_path), 0)
            self.assertEqual(benchmark.stage_main("broken", context_path), 1)
        good = json.loads((self.folder / "results" / "good.json").read_text(encoding="utf-8"))
        self.assertEqual((good["tool_seconds"], good["outputs"]), (1.5, {"x": "y"}))
        self.assertIn("gpu_peak_bytes", good)
        failed = json.loads((self.folder / "results" / "broken.json").read_text(encoding="utf-8"))
        self.assertEqual((failed["returncode"], failed["error"]), (1, "RuntimeError: broken"))


MACHINE = {
    "hardware": {"cpu": "Test CPU", "logical_cpus": 8, "physical_cores": 4, "ram_bytes": 16 << 30,
                 "devices": [{"spec": "cpu", "name": "CPU", "backend": "cpu"}], "gpus": [], "os": "TestOS"},
    "software": {"python": "3.13", "pytorch": "2.12", "packages": {"numpy": "2.5.3", "esm": None}},
    "conditions": {"cpu_load_percent": 3.0, "available_ram_bytes": 8 << 30, "on_battery": None,
                   "other_emapssn_processes": [], "caches": {"numba_cache_files": 4, "gpu_kernel_cache_files": 0}},
}
ALIGNMENT_DECISIONS = [
    {"kind": "alignment_plan", "unit": "pairs/s", "ranking": [1, 0], "winner": 1, "candidates": [
        {"device": "CPU", "backend": "cpu", "variant": "scalar", "lanes": 1, "value": 500.0, "error": None},
        {"device": "Test GPU", "backend": "cuda", "variant": "tiled", "lanes": 2, "value": 5000.0, "error": None},
        {"device": "Test GPU", "backend": "cuda", "variant": "scalar", "lanes": 8, "value": None, "error": "out of memory"},
    ]},
    {"kind": "matmul_precision", "choice": "ieee_fp32", "reason": "too_little_speedup", "unit": "pairs/s",
     "rates": [{"variant": "tiled", "precision": "ieee_fp32", "value": 4000.0},
               {"variant": "tiled", "precision": "tf32", "value": 4100.0}]},
]


class BenchmarkRunTests(unittest.TestCase):
    """Whole runs with stand-in steps: the order, skips, failures, interruptions and reports."""

    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.dict(os.environ, {benchmark.WORK_DIR_VARIABLE: str(self.folder)}))
        # The machine's own checks (BLAST+, packages) stay out of these runs; the model's stays in.
        stages = tuple(stage if stage.ready is benchmark.model_problem else replace(stage, ready=None)
                       for stage in benchmark.STAGES)
        for name, value in (("STAGES", stages), ("STAGES_BY_KEY", {stage.key: stage for stage in stages}),
                            ("STAGES_BY_NUMBER", {stage.number: stage for stage in stages})):
            self.enterContext(mock.patch.object(benchmark, name, value))
        self.enterContext(mock.patch.object(benchmark, "describe_machine", return_value=MACHINE))
        self.enterContext(mock.patch.object(benchmark, "program_version",
                                            return_value={"version": "0.3.0", "commit": "abc", "dirty": False}))
        self.cached = mock.patch.object(benchmark, "reference_model_cached", return_value=True)
        self.enterContext(self.cached)
        self.enterContext(mock.patch.object(benchmark, "run_child", self.stand_in_step))
        self.steps, self.failing, self.interrupt_at, self.download_error = [], set(), None, None

    def stand_in_step(self, step, context, temp, *, number=0, record=None, echo=True, offline=False):
        self.steps.append(step)
        log = temp / "logs" / f"stage_{number:02d}_{step}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"output of {step}\n", encoding="utf-8")
        if record is not None and step == "alignment":
            with open(record, "a", encoding="utf-8") as handle:
                handle.writelines(json.dumps(decision) + "\n" for decision in ALIGNMENT_DECISIONS)
        interrupted = step == self.interrupt_at
        failed = step in self.failing or (step == "download_model" and self.download_error)
        run = benchmark.StageRun(returncode=None if interrupted else int(bool(failed)), wall_seconds=1.5,
                                 cpu_seconds=1.0, peak_rss_bytes=1 << 20, peak_processes=1,
                                 interrupted=interrupted, log=log)
        child = {} if interrupted else {"returncode": int(bool(failed)), "tool_seconds": 1.0, "import_seconds": 0.25}
        if step == "download_model" and self.download_error:
            child["error"] = self.download_error
        return run, child

    def run_benchmark(self, **arguments):
        with redirect_stdout(StringIO()) as shown:
            code = benchmark.run_benchmark(result_path=self.folder / "result.json", echo=False, **arguments)
        return code, shown.getvalue(), json.loads((self.folder / "result.json").read_text(encoding="utf-8"))

    def statuses(self, result):
        return {stage["key"]: stage["status"] for stage in result["stages"]}

    def test_a_full_run_writes_both_reports_and_leaves_no_temporary_files(self):
        code, shown, result = self.run_benchmark()
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.steps, [step for stage in benchmark.STAGES for step in stage.steps])
        self.assertEqual(set(self.statuses(result).values()), {"completed"})
        text_path, json_path = Path(result["report_text"]), Path(result["report_json"])
        self.assertEqual((text_path.parent, json_path.parent), (self.folder, self.folder))
        self.assertFalse((self.folder / "temp").exists())
        text = text_path.read_text(encoding="utf-8")
        for stage in benchmark.STAGES:
            self.assertIn(str(stage.title), text)
        self.assertIn("Test GPU, tiled plan, 2 lanes", text)
        self.assertIn("Failed (out of memory)", text)
        self.assertIn("Matrix-product precision FP32, because TF32 was less than 1.10 times as fast.", text)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["stages"][2]["auto_choices"][0], {"kind": "alignment_plan",
                                                              "choice": "Test GPU, tiled plan, 2 lanes"})
        self.assertTrue(shown.rstrip().endswith(f"Its data are saved as {json_path}"))
        self.assertIn(f"The report is saved as {text_path}", shown)
        self.assertIn("EMAP-SSN benchmark report", shown)

    def test_a_failed_stage_skips_the_stages_that_need_it_while_the_others_run(self):
        self.failing = {"alignment"}
        code, _shown, result = self.run_benchmark()
        self.assertEqual((code, result["status"]), (1, "failed"))
        statuses = self.statuses(result)
        self.assertEqual(statuses["alignment"], "failed")
        for key in ("ssn_layout", "umap_layout", "clustering", "injection", "msa"):
            self.assertEqual(statuses[key], "skipped", key)
        for key in ("sanitize", "embeddings", "search", "blast"):
            self.assertEqual(statuses[key], "completed", key)
        stages = {stage["key"]: stage for stage in result["stages"]}
        self.assertIn("stage 3", stages["msa"]["reason"])
        data = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
        self.assertEqual(data["stages"][2]["log_tail"], ["output of alignment"])

    def test_an_interrupted_run_still_writes_its_report(self):
        self.interrupt_at = "ssn_layout"
        code, _shown, result = self.run_benchmark()
        self.assertEqual((code, result["status"]), (1, "interrupted"))
        statuses = self.statuses(result)
        self.assertEqual(statuses["ssn_layout"], "interrupted")
        self.assertEqual({statuses[stage.key] for stage in benchmark.STAGES[4:]}, {"not_run"})
        self.assertTrue(Path(result["report_text"]).is_file())
        self.assertFalse((self.folder / "temp").exists())

    def test_only_the_chosen_stages_and_what_they_need_run(self):
        code, _shown, result = self.run_benchmark(stages="4")
        self.assertEqual(code, 0)
        self.assertEqual(self.steps, ["sanitize", "embeddings", "alignment", "ssn_layout"])
        self.assertEqual({key for key, status in self.statuses(result).items() if status == "not_selected"},
                         {"umap_layout", "clustering", "search", "injection", "msa", "blast"})

    def test_a_second_run_keeps_the_first_report(self):
        _code, _shown, first = self.run_benchmark()
        first_text = Path(first["report_text"]).read_bytes()
        _code, _shown, second = self.run_benchmark()
        self.assertNotEqual(first["report_text"], second["report_text"])
        self.assertEqual(Path(first["report_text"]).read_bytes(), first_text)
        self.assertEqual(len(list(self.folder.glob("Benchmark_Report_*.txt"))), 2)

    def test_a_running_benchmark_stops_a_second_one_with_exit_code_2(self):
        lock = benchmark.acquire_lock(self.folder / "temp")
        self.addCleanup(benchmark.release_lock, lock)
        code, shown, result = self.run_benchmark()
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "could_not_start")
        self.assertIn("Another benchmark is running", shown)
        self.assertEqual(self.steps, [])
        self.assertTrue(lock.exists())

    def test_a_model_that_can_t_be_downloaded_skips_the_stages_that_embed(self):
        self.download_error = "OSError: offline"
        with mock.patch.object(benchmark, "reference_model_cached", return_value=False):
            code, _shown, result = self.run_benchmark()
        self.assertEqual(code, 0, "a stage this machine can't run is skipped, not failed")
        self.assertEqual(self.steps, ["download_model", "sanitize", "blast"])
        stages = {stage["key"]: stage for stage in result["stages"]}
        self.assertIn("could not be downloaded (OSError: offline)", stages["embeddings"]["reason"])
        self.assertEqual(stages["alignment"]["status"], "skipped")


class BenchmarkReportTests(unittest.TestCase):
    """The .txt lines up its columns in any script."""

    def test_wide_characters_take_two_columns(self):
        self.assertEqual(benchmark.display_width("abc"), 3)
        self.assertEqual(benchmark.display_width("酶活性"), 6)
        self.assertEqual(benchmark.display_width("é"), 1)

    def test_table_columns_line_up_by_display_width(self):
        lines = benchmark.table([["酶", "x", "end"], ["abc", "yy", "end"]])
        self.assertEqual([benchmark.display_width(line[:line.index("end")]) for line in lines], [9, 9])

    def test_a_plan_names_its_device_lanes_and_memory_profile(self):
        tiled = {"device": "GPU", "backend": "cuda", "variant": "tiled", "lanes": 2, "profile": "balanced"}
        scalar = {"device": "GPU", "backend": "cuda", "variant": "scalar", "lanes": 1, "profile": None}
        cpu = {"device": "CPU", "backend": "cpu", "variant": "scalar", "lanes": 1}
        self.assertEqual(str(benchmark.candidate_text(tiled, "alignment_plan")),
                         "GPU, tiled plan with the balanced memory profile, 2 lanes")
        self.assertEqual(str(benchmark.candidate_text(scalar, "injection_plan")), "GPU, scalar plan, 1 lane")
        self.assertEqual(str(benchmark.candidate_text(cpu, "alignment_plan")), "CPU, scalar plan")
        self.assertEqual(benchmark.candidate_text(tiled, "embedding_device"), "GPU")

    def test_a_table_that_a_tie_reordered_states_the_tie_rule(self):
        # rank_benchmark_results gives a near-tie to fewer lanes, as stage 8 of a real run showed.
        decision = {"kind": "injection_plan", "unit": "pairs/s", "direction": "higher", "tie_fraction": 0.03,
                    "ranking": [0, 1], "winner": 0, "candidates": [
                        {"device": "GPU", "backend": "cuda", "variant": "tiled", "lanes": 2, "value": 15769.0},
                        {"device": "GPU", "backend": "cuda", "variant": "tiled", "lanes": 8, "value": 16096.0},
                    ]}
        blocks = benchmark.decision_blocks(decision)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(str(blocks[-1][1]), "Results within 3 % of the best count as a tie, which goes to the CPU, "
                                             "a scalar plan, less peak memory and fewer lanes, in that order.")
        self.assertEqual(len(benchmark.decision_blocks(dict(decision, ranking=[1, 0], winner=1))), 2,
                         "a ranking in measured order needs no rule")
        self.assertEqual(len(benchmark.decision_blocks(dict(decision, tie_fraction=None))), 2,
                         "a record without the tie says nothing about it")

    def test_for_times_a_faster_candidate_ranked_lower_states_the_tie_rule(self):
        decision = {"kind": "layout_device", "unit": "s", "direction": "lower", "tie_fraction": 0.05,
                    "ranking": [0, 1], "winner": 0, "candidates": [
                        {"device": "CPU", "backend": "cpu", "value": 0.44},
                        {"device": "GPU", "backend": "cuda", "value": 1.31},
                    ]}
        self.assertEqual(len(benchmark.decision_blocks(decision)), 2)
        decision["candidates"][1]["value"] = 0.43
        self.assertIn("within 5 % of the best", str(benchmark.decision_blocks(decision)[-1][1]))

    def test_the_auto_trials_are_told_apart_from_the_work_after_them(self):
        outcome = benchmark.StageResult(benchmark.STAGES_BY_KEY["alignment"], status="completed")
        outcome.decisions = [{"kind": "matmul_precision", "time": 1003.0}, {"kind": "alignment_plan", "time": 1010.0},
                             {"kind": "host_cache", "time": 1012.0}]
        outcome.child = {"tool_seconds": 15.0,
                         "steps": {"alignment": {"started_at": 1000.0, "finished_at": 1015.0}}}
        self.assertEqual(benchmark.trial_split(outcome), (10.0, 5.0))
        inputs = {"workload": {"main": {"sequences": 4, "residues": 40, "pairs": 6, "cells": 600}}}
        rates = benchmark.stage_rates(outcome, inputs)
        self.assertEqual(rates[-1], {"value": 6 / 5.0, "unit": "pairs/s after the Auto trials"})
        self.assertEqual(rates[0], {"value": 6 / 15.0, "unit": "pairs/s"})
        outcome.decisions = []
        self.assertIsNone(benchmark.trial_split(outcome))

    def test_a_translated_report_keeps_its_columns_lined_up(self):
        previous = Localization.set_translator(lambda template, n: "测试" + Localization.english_text(template, n))
        self.addCleanup(Localization.set_translator, previous)
        started = datetime(2026, 10, 9, 21, 0, 0).astimezone()
        outcomes = [benchmark.StageResult(stage, status="skipped", reason=benchmark.Message("Not here."))
                    for stage in benchmark.STAGES]
        run = benchmark.BenchmarkRun(started=started, selected=[], outcomes=outcomes,
                                     finished=started + timedelta(seconds=90), machine=MACHINE)
        text = benchmark.render(benchmark.report_blocks(run), Localization.display_text)
        summary = text.split("\n测试Summary\n")[1].split("\n\n")[0].splitlines()[1:]
        self.assertEqual(len(summary), 11)
        starts = {benchmark.display_width(line[:line.index("测试Skipped" if index else "测试Status")])
                  for index, line in enumerate(summary)}
        self.assertEqual(len(starts), 1, summary)


def qt_application():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def every_kind_of_stage():
    """A run whose stages end every way, with every kind of Auto decision, and data without letters."""
    started = datetime(2026, 10, 9, 21, 0, 0).astimezone()
    stages = benchmark.STAGES
    measured = benchmark.StageRun(returncode=0, wall_seconds=20.0, cpu_seconds=30.0,
                                  peak_rss_bytes=1 << 30, peak_processes=3)
    child = {"tool_seconds": 15.0, "import_seconds": 1.0, "gpu_peak_bytes": 1 << 20,
             "steps": {"alignment": {"started_at": 1000.0, "finished_at": 1015.0}},
             "details": {"nodes": 4, "edges": 6, "tree_seconds": 1.5, "merge_seconds": 0.5}}
    # #4 measured better than the winner, #2, but lost the tie, so the tie rule shows too.
    ranking = {"unit": "pairs/s", "direction": "higher", "tie_fraction": 0.03, "ranking": [1, 3, 0], "winner": 1,
               "time": 1010.0, "size_class": "small", "candidates": [
        {"device": "#1", "backend": "cpu", "variant": "1", "lanes": 1, "value": 5.0, "error": None},
        {"device": "#2", "backend": "cuda", "variant": "2", "lanes": 2, "value": 50.0, "error": None,
         "profile": "3", "peak_memory_bytes": 1 << 20},
        {"device": "#3", "backend": "cuda", "variant": "2", "lanes": 8, "value": None, "error": "4"},
        {"device": "#4", "backend": "cuda", "variant": "2", "lanes": 4, "value": 51.0, "error": None},
    ]}
    decisions = [dict(ranking, kind=kind) for kind in benchmark.DECISION_TITLES] + [
        {"kind": "matmul_precision", "choice": "ieee_fp32", "reason": reason, "unit": "pairs/s", "time": 1003.0,
         "rates": [{"variant": "2", "precision": "ieee_fp32", "value": 4.0}]}
        for reason in benchmark.PRECISION_REASONS
    ] + [{"kind": "host_cache", "choice": choice, "limit_bytes": 1 << 30, "embedding_bytes": 1 << 20}
         for choice in benchmark.HOST_CACHE_TEXT]
    outcomes = []
    for stage, status in zip(stages, ["completed", "completed", "completed", "failed", "skipped",
                                      "interrupted", "not_run", "completed", "completed", "not_selected"]):
        outcome = benchmark.StageResult(stage, status=status)
        if status in ("completed", "failed", "interrupted"):
            outcome.run, outcome.child = measured, dict(child)
        if stage.key == "alignment":
            outcome.decisions = decisions
        if status == "failed":
            outcome.reason = benchmark.Message("The stage stopped with exit code {code}.", code=1)
            outcome.log_tail = ["12 34"]
        if status == "skipped":
            outcome.reason = benchmark.umap_problem({}) or benchmark.Message("It needs {stages}, which did not complete.",
                                                                              stages="3")
        outcomes.append(outcome)
    machine = {
        "hardware": {"cpu": "-", "logical_cpus": 8, "physical_cores": 4, "ram_bytes": 1 << 34, "gpus": [
            {"name": "#2", "driver": "1.0", "compute_capability": "8.9"}],
            "devices": [{"spec": "1", "name": "#2", "backend": "cuda", "memory_bytes": 1 << 33}], "os": "-"},
        "software": {"python": "3.13", "pytorch": "2.12", "packages": {}, "numba": {"threads": 6, "usable_cpus": 8}},
        "conditions": {"cpu_load_percent": 50.0, "available_ram_bytes": 1 << 33, "on_battery": True,
                       "other_emapssn_processes": ["1"], "caches": {"numba_cache_files": 1, "gpu_kernel_cache_files": 2}},
    }
    inputs = {"dataset": {"name": "-"}, "workload": {
        "main": {"sequences": 4, "residues": 40, "pairs": 6, "cells": 600},
        "injection": {"sequences": 2, "residues": 20, "pairs": 9, "cells": 900},
        "msa": {"sequences": 3, "residues": 30, "pairs": 3, "cells": 300},
    }}
    return benchmark.BenchmarkRun(
        started=started, selected=[stage.key for stage in stages[:-1]], outcomes=outcomes, limit=20,
        finished=started + timedelta(seconds=90), inputs=inputs, machine=machine, interrupted=True,
        error="1", program={"version": "0.3.0", "commit": "0593ae3", "dirty": True},
        model={"download_seconds": 2.0}, language="pseudo", temp_peak_bytes=1 << 30,
    )


class BenchmarkLanguageTests(unittest.TestCase):
    """The .txt follows the program's language; the terminal, the .json and MCP stay English."""

    def test_every_text_of_the_report_comes_from_the_catalog(self):
        from tests.translation_fixtures import outside_the_catalog, pseudo_language

        pseudo_language(self, qt_application())
        text = benchmark.render(benchmark.report_blocks(every_kind_of_stage()), Localization.display_text)
        # Names the report shows as they are: the model, two programs, a precision and the language's
        # code; and the units after numbers, which every language writes alike.
        for name in (f"{benchmark.REFERENCE_MODEL} ({benchmark.MODEL_REPOSITORY})", "Python", "PyTorch",
                     "FP32", "pseudo"):
            text = text.replace(name, "")
        text = re.sub(r"\d[\d,.]* (?:ms|s|B|KiB|MiB|GiB|TiB)\b", "", text)
        self.assertEqual(outside_the_catalog(text), "")

    def test_a_layout_decision_names_its_size_class_in_the_reports_language(self):
        # A value filled into a translated text shows inside the pseudo-language's
        # brackets, where the check above can't tell it from the text: check it here.
        from tests.translation_fixtures import pseudo_language

        pseudo_language(self, qt_application())
        for size in ("small", "medium", "massive"):
            with self.subTest(size=size):
                title = benchmark.decision_blocks({"kind": "layout_device", "size_class": size})[0][1]
                self.assertIn(f" {Localization.pseudo_translate(size)} ", Localization.display_text(title))

    def test_a_plan_names_its_variant_and_memory_profile_in_the_reports_language(self):
        from tests.translation_fixtures import pseudo_language

        pseudo_language(self, qt_application())
        for variant in benchmark.PLAN_NAMES:
            with self.subTest(variant=variant):
                shown = benchmark.candidate_text({"device": "#1", "backend": "cpu", "variant": variant}, "search_plan")
                self.assertIn(f" {Localization.pseudo_translate(variant)} ", Localization.display_text(shown))
        for profile in benchmark.PROFILE_NAMES:
            with self.subTest(profile=profile):
                candidate = {"device": "#1", "backend": "cuda", "variant": "tiled", "profile": profile, "lanes": 2}
                shown = Localization.display_text(benchmark.candidate_text(candidate, "alignment_plan"))
                for name in ("tiled", profile):
                    self.assertIn(f" {Localization.pseudo_translate(name)} ", shown)
        precision = {"kind": "matmul_precision", "reason": "not_equivalent", "unit": "pairs/s",
                     "rates": [{"precision": "ieee_fp32", "variant": "tiled", "value": 1.0}]}
        rows = benchmark.decision_blocks(precision)[1][1]
        self.assertEqual(Localization.display_text(rows[1][1]), Localization.pseudo_translate("tiled"))

    def test_a_run_in_the_pseudo_language_writes_only_the_txt_in_it(self):
        qt_application()
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with mock.patch.dict(os.environ, {benchmark.WORK_DIR_VARIABLE: str(folder), "SSN_PSEUDO_TRANSLATION": "1"}), \
                mock.patch.object(benchmark, "describe_machine", return_value=MACHINE), \
                mock.patch.object(benchmark, "reference_model_cached", return_value=True), \
                mock.patch.object(benchmark, "run_child", BenchmarkRunTests.stand_in_step.__get__(self)), \
                redirect_stdout(StringIO()) as shown:
            self.steps, self.failing, self.interrupt_at, self.download_error = [], set(), None, None
            code = benchmark.run_benchmark(stages="1", result_path=folder / "result.json", echo=False)
        self.assertEqual(code, 0)
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        text = Path(result["report_text"]).read_text(encoding="utf-8")
        self.assertEqual(text.splitlines()[0], Localization.pseudo_translate("EMAP-SSN benchmark report"))
        shown = shown.getvalue()
        self.assertIn("\nEMAP-SSN benchmark report\n", shown)
        data = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
        self.assertEqual((data["language"], data["stages"][0]["title"], data["stages"][0]["status"]),
                         ("pseudo", "Sanitize sequences", "completed"))
        pseudo_letters = set(Localization.pseudo_translate("abcdefghijklmnopqrstuvwxyz")) - set("[]")
        for english in (shown, json.dumps(data, ensure_ascii=False)):
            self.assertEqual(pseudo_letters & set(english), set())
        self.assertEqual(str(benchmark.STATUS_TEXT["completed"]), Localization.display_text(benchmark.STATUS_TEXT["completed"]),
                         "the language is removed once the .txt is written")

    def test_a_language_that_can_t_be_installed_leaves_the_report_english(self):
        import desktop.Desktop_App as Desktop_App

        qt_application()
        run = every_kind_of_stage()
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with mock.patch.object(Desktop_App, "install_translations", side_effect=LookupError("no catalog")), \
                redirect_stdout(StringIO()) as shown:
            benchmark.write_reports(run, folder)
        self.assertTrue(run.text_path.read_text(encoding="utf-8").startswith("EMAP-SSN benchmark report\n"))
        self.assertIn("could not be installed (LookupError: no catalog)", shown.getvalue())

    def test_a_language_setting_that_can_t_be_read_leaves_the_report_english(self):
        import desktop.Desktop_App as Desktop_App

        with mock.patch.object(Desktop_App, "startup_language", side_effect=OSError("unreadable")), \
                redirect_stdout(StringIO()) as shown:
            self.assertIsNone(benchmark.report_language())
        self.assertIn("unreadable", shown.getvalue())


class BenchmarkToolsCardTests(unittest.TestCase):
    """The Benchmark card at the end of the Tools window's Manual Tools tab."""

    def open_window(self):
        import EMAPSSN_Tools
        from PySide6.QtWidgets import QTextBrowser
        from tests.tools_gui_fixtures import isolated_tools_project

        qt_application()
        isolated_tools_project(self)
        with mock.patch.object(EMAPSSN_Tools, "ResponsiveTextBrowser", QTextBrowser):
            window = EMAPSSN_Tools.ToolsGUI()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        return EMAPSSN_Tools, window

    def benchmark_card(self, window):
        from PySide6.QtWidgets import QFrame

        for index in range(window.tabs.count()):
            page = window.tabs.widget(index)
            if page.property("descriptionKey") == "Others":
                return page.widget().findChildren(QFrame, "toolSectionCard")[-1]
        self.fail("The Tools window has no Manual Tools tab.")

    def test_the_manual_tools_tab_ends_with_the_benchmark_card(self):
        from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QWidget

        tools, window = self.open_window()
        card = self.benchmark_card(window)
        self.assertEqual(card.findChild(QLabel, "toolTitle").text(), "⏱️ Benchmark")
        actions = card.findChild(QWidget, "toolActionButtons").findChildren(QPushButton)
        self.assertEqual([button.text() for button in actions], ["Run Benchmark"])
        self.assertIn(card.layout(), window._tool_form_layouts, "it shares the cards' left section")
        folder = card.findChild(QLineEdit)
        self.assertTrue(folder.isReadOnly())
        self.assertEqual(Path(folder.text()), BENCHMARK_DIR)
        self.assertEqual(Path(tools.BENCHMARK_SCRIPT), BENCHMARK_DIR / "Run_Benchmark.py")
        self.assertNotIn("Run_Benchmark.py", tools.get_tool_titles(), "the help heading names no script")

    def test_run_benchmark_asks_then_starts_the_benchmark_in_a_console(self):
        from PySide6.QtWidgets import QPushButton

        tools, window = self.open_window()
        button = self.benchmark_card(window).findChild(QPushButton, "runBenchmarkButton")
        answers = tools.QMessageBox.StandardButton
        with mock.patch.object(tools.QMessageBox, "question", return_value=answers.No) as asked, \
                mock.patch.object(tools, "launch_in_terminal") as launch:
            button.click()
        asked.assert_called_once()
        launch.assert_not_called()
        with mock.patch.object(tools.QMessageBox, "question", return_value=answers.Yes), \
                mock.patch.object(tools, "launch_in_terminal") as launch:
            button.click()
        launch.assert_called_once_with(
            [sys.executable, "-u", tools.BENCHMARK_SCRIPT], cwd=tools.BENCHMARK_DIR,
            hold=tools.HoldMode.ALWAYS, title="Run_Benchmark.py",
        )

    def test_a_benchmark_that_can_t_start_is_reported(self):
        tools, window = self.open_window()
        answers = tools.QMessageBox.StandardButton
        with mock.patch.object(tools.QMessageBox, "question", return_value=answers.Yes), \
                mock.patch.object(tools, "launch_in_terminal", side_effect=OSError("no console")), \
                mock.patch.object(tools.QMessageBox, "critical") as reported:
            window.run_benchmark()
        reported.assert_called_once()
        self.assertIn("no console", reported.call_args.args[2])
        self.assertIn(tools.BENCHMARK_SCRIPT, reported.call_args.args[2])

    def test_the_folder_button_opens_the_benchmark_folder(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QPushButton

        _tools, window = self.open_window()
        button = self.benchmark_card(window).findChild(QPushButton, "openBenchmarkFolderButton")
        with mock.patch.object(QDesktopServices, "openUrl") as opened:
            button.click()
        opened.assert_called_once_with(QUrl.fromLocalFile(str(BENCHMARK_DIR)))

    def test_the_question_before_a_run_shows_in_the_window_s_language(self):
        from tests.translation_fixtures import outside_the_catalog, pseudo_language

        tools, window = self.open_window()
        pseudo_language(self, qt_application())
        with mock.patch.object(tools.QMessageBox, "question",
                               return_value=tools.QMessageBox.StandardButton.No) as asked:
            window.run_benchmark()
        _parent, title, text = asked.call_args.args[:3]
        self.assertEqual((outside_the_catalog(title), outside_the_catalog(text)), ("", ""))


STAND_IN_BENCHMARK = '''
import json, os, pathlib, sys, time

arguments = sys.argv[1:]
behaviour = json.loads(pathlib.Path(__file__).with_name("behaviour.json").read_text(encoding="utf-8"))
folder = pathlib.Path(os.environ["SSN_BENCHMARK_WORK_DIR"])
text, data = folder / "Benchmark_Report_2026-10-09_21-00-00.txt", folder / "Benchmark_Report_2026-10-09_21-00-00.json"
print("benchmark arguments", json.dumps(arguments), flush=True)
if behaviour["exit"] != 2:
    text.write_text("report", encoding="utf-8")
    data.write_text("{}", encoding="utf-8")
summary = dict(behaviour["summary"], report_text=str(text), report_json=str(data),
               stages_argument=arguments[arguments.index("--stages") + 1] if "--stages" in arguments else None)
if behaviour.get("write_result", True):
    pathlib.Path(arguments[arguments.index("--result") + 1]).write_text(json.dumps(summary), encoding="utf-8")
time.sleep(behaviour.get("sleep", 0))  # A cancelled job's result is ignored even once written.
raise SystemExit(behaviour["exit"])
'''


class BenchmarkMCPJobTests(unittest.IsolatedAsyncioTestCase):
    """emapssn_pipeline(action="start_benchmark") jobs, with a stand-in Run_Benchmark.py."""

    async def asyncSetUp(self):
        from mcp_server.pipeline.Pipeline_Jobs import PipelineJobManager

        self.temporary = Path(self.enterContext(tempfile.TemporaryDirectory(ignore_cleanup_errors=True)))
        self.project = self.temporary / "project"
        script = self.project / "src" / "resources" / "benchmark" / "Run_Benchmark.py"
        script.parent.mkdir(parents=True)
        script.write_text(STAND_IN_BENCHMARK, encoding="utf-8")
        self.behaviour = script.with_name("behaviour.json")
        self.reports = self.temporary / "reports"
        self.reports.mkdir()
        self.enterContext(mock.patch.dict(os.environ, {"SSN_BENCHMARK_WORK_DIR": str(self.reports)}))
        self.manager = PipelineJobManager(self.project, max_pending=2, history_limit=4, termination_grace=0.5,
                                          temporary_parent=self.temporary)
        await self.manager.start()
        self.addAsyncCleanup(self.manager.close)

    async def run_job(self, behaviour, stages=None):
        self.behaviour.write_text(json.dumps(behaviour), encoding="utf-8")
        submitted = await self.manager.submit_benchmark_job(stages=stages)
        self.assertEqual(submitted["tool_id"], "run_benchmark")
        return await self.manager.wait_for_terminal(submitted["job_id"])

    async def test_a_finished_benchmark_returns_its_summary_and_report_files(self):
        job = await self.run_job({"exit": 0, "summary": {"status": "completed", "stages": []}}, stages="3,search")
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual((job["result"]["status"], job["result"]["stages_argument"]), ("completed", "3,search"))
        self.assertEqual(job["output_locations"]["BENCHMARK_REPORT"], str(self.reports / "Benchmark_Report_2026-10-09_21-00-00.txt"))
        self.assertEqual(job["output_locations"]["BENCHMARK_REPORT_DIR"], str(self.reports))
        self.assertEqual(sorted(Path(item["path"]).name for item in job["output_files"]),
                         ["Benchmark_Report_2026-10-09_21-00-00.json", "Benchmark_Report_2026-10-09_21-00-00.txt"])
        self.assertEqual({item["change"] for item in job["output_files"]}, {"created"})

    async def test_every_stage_runs_without_a_stages_argument(self):
        job = await self.run_job({"exit": 0, "summary": {"status": "completed"}})
        self.assertIsNone(job["result"]["stages_argument"])

    async def test_a_failed_stage_fails_the_job_but_keeps_the_result(self):
        stages = [{"number": 3, "key": "alignment", "status": "failed"}, {"number": 1, "key": "sanitize", "status": "completed"}]
        job = await self.run_job({"exit": 1, "summary": {"status": "failed", "stages": stages}})
        self.assertEqual(job["status"], "failed")
        self.assertIn("stage(s) 3 alignment failed", job["failure_message"])
        self.assertEqual(job["result"]["stages"], stages)

    async def test_a_benchmark_that_could_not_start_says_why(self):
        job = await self.run_job({"exit": 2, "summary": {"status": "could_not_start",
                                                         "reason": "Another benchmark is running."}})
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["failure_message"], "The benchmark could not start: Another benchmark is running.")

    async def test_a_missing_result_after_success_fails_the_job(self):
        job = await self.run_job({"exit": 0, "write_result": False, "summary": {}})
        self.assertEqual(job["status"], "failed")
        self.assertIn("The benchmark process exited with code 0, but its result could not be read",
                      job["failure_message"])
        self.assertIsNone(job["result"])

    async def test_a_cancelled_benchmark_has_no_result(self):
        self.behaviour.write_text(json.dumps({"exit": 0, "sleep": 30, "summary": {"status": "completed"}}),
                                  encoding="utf-8")
        submitted = await self.manager.submit_benchmark_job()
        for _attempt in range(500):
            if (await self.manager.get_job(submitted["job_id"]))["status"] == "running":
                break
            await asyncio.sleep(0.01)
        await self.manager.cancel(submitted["job_id"])
        job = await self.manager.wait_for_terminal(submitted["job_id"])
        self.assertEqual(job["status"], "cancelled")
        self.assertIsNone(job["result"])


class BenchmarkMCPActionTests(unittest.IsolatedAsyncioTestCase):
    """The action's arguments and its entry in the pipeline catalog."""

    def setUp(self):
        from mcp_server.pipeline import Pipeline_Operations

        self.operations = Pipeline_Operations
        self.submit = mock.AsyncMock(return_value={
            "job_id": "job", "tool_id": "run_benchmark", "status": "queued", "queue_position": 1,
            "created_at": "now", "started_at": None, "finished_at": None, "exit_code": None,
            "failure_message": None, "cancellation_requested": False, "settings_snapshot": "snapshot.json",
            "stdout_log": "out", "stderr_log": "err", "output_locations": {},
        })
        context = mock.Mock(jobs=mock.Mock(submit_benchmark_job=self.submit))
        self.enterContext(mock.patch.object(Pipeline_Operations, "_context", return_value=context))

    async def test_stages_go_to_the_script_by_number_or_name(self):
        info = await self.operations.start_benchmark(mock.Mock(), stages=[3, " Search ", "10"])
        self.submit.assert_awaited_once_with(stages="3,search,10")
        self.assertEqual(info.tool_id, "run_benchmark")
        await self.operations.start_benchmark(mock.Mock())
        self.submit.assert_awaited_with(stages=None)

    async def test_a_stage_that_does_not_exist_is_refused_before_queueing(self):
        from mcp.server.mcpserver.exceptions import ToolError

        for stages in ([11], ["nope"], []):
            with self.subTest(stages=stages), self.assertRaises(ToolError):
                await self.operations.start_benchmark(mock.Mock(), stages=stages)
        self.submit.assert_not_awaited()

    def test_the_catalog_names_the_script_s_stages(self):
        catalog = self.operations.list_pipeline_tools()
        self.assertEqual(catalog.benchmark["stages"], [f"{stage.number} {stage.key}" for stage in benchmark.STAGES])
        self.assertEqual(catalog.benchmark["job_tool_id"], "run_benchmark")
        self.assertEqual(len(catalog.tools), 14, "the benchmark is no pipeline tool")


@unittest.skipUnless(
    os.environ.get("EMAPSSN_BENCHMARK_SMOKE_TEST") == "1",
    "runs the real benchmark on 20 sequences for about two minutes; set EMAPSSN_BENCHMARK_SMOKE_TEST=1",
)
class BenchmarkSmokeTests(unittest.TestCase):
    def test_a_real_run_on_20_sequences_completes_every_stage_this_machine_can_run(self):
        if not benchmark.reference_model_cached():
            self.skipTest("ESM-2 8M is not in the Hugging Face cache, and tests never download")
        with tempfile.TemporaryDirectory() as folder:
            result_path = Path(folder) / "result.json"
            completed = subprocess.run(
                [sys.executable, "-u", str(BENCHMARK_DIR / "Run_Benchmark.py"), "--limit", "20",
                 "--result", str(result_path)],
                env={**os.environ, benchmark.WORK_DIR_VARIABLE: folder}, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=1800,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout[-4000:])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertFalse((Path(folder) / "temp").exists())
        self.assertEqual({stage["status"] for stage in result["stages"]} - {"skipped"}, {"completed"})


if __name__ == "__main__":
    unittest.main()
