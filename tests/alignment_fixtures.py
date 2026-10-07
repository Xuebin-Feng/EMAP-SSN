"""Fakes shared by the alignment, Network Injection and benchmark-trial tests.

ImmediateExecutor stands in for concurrent.futures.ThreadPoolExecutor: it runs
every submitted call at once and records each instance and submission.
RecordingTrial is a real BenchmarkTrial whose clock never advances, so every
pair is submitted before the deadline, and which logs "trial-start" and
"trial-stop" into an event list shared with the fakes of a test.
BatchResumeScanFixture writes one-pair batch files for the resume scans
(scan_existing_batches) and reads back the report of a backed-up batch folder.
"""
import os
import sys
from concurrent.futures import Future

import h5py
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Embedding_Alignment_Engine as alignment_engine  # noqa: E402


class ImmediateExecutor:
    instances = []

    def __init__(self, max_workers, **kwargs):
        self.max_workers = max_workers
        self.options = kwargs
        self.submissions = []
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def submit(self, function, *args):
        self.submissions.append((function, args))
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as error:
            future.set_exception(error)
        return future


class RecordingTrial(alignment_engine.BenchmarkTrial):
    """Benchmark trial with a frozen clock that logs its phase boundaries."""

    def __init__(self, events):
        super().__init__(clock=lambda: 0.0)
        self.events = events

    def start(self):
        super().start()
        self.events.append("trial-start")

    def stop(self, total_pairs):
        super().stop(total_pairs)
        self.events.append("trial-stop")


class BatchResumeScanFixture:
    """One-pair batch files and the backup report of a resume scan.

    A test class sets MATCHING_ATTRS to the batch attributes its tool's scan
    accepts for the run under test; write_batch() overrides them by keyword.
    """

    MATCHING_ATTRS = {}
    REPORT_NAME = "batch_attributes_info.txt"

    def write_batch(self, folder, name="batch_00000.h5", *, omit=(), j=(1,), **attrs):
        """Write pair (0, 1) with every result column except those in omit."""
        os.makedirs(folder, exist_ok=True)
        columns = {
            "i": np.asarray([0], np.uint32),
            "j": np.asarray(j, np.uint32),
            "l_score": np.asarray([1.0], np.float32),
            "l_len": np.asarray([1], np.uint16),
            "g_score": np.asarray([2.0], np.float32),
            "g_len": np.asarray([1], np.uint16),
        }
        with h5py.File(os.path.join(folder, name), "w") as hf:
            for key, value in {**self.MATCHING_ATTRS, **attrs}.items():
                hf.attrs[key] = value
            for key, data in columns.items():
                if key not in omit:
                    hf.create_dataset(key, data=data)

    def assert_backed_up(self, results_dir, backup_dir, batches=("batch_00000.h5",)):
        """Check the folder moved whole and return the report's reason lines."""
        self.assertEqual(os.listdir(results_dir), [])
        self.assertEqual(
            sorted(os.listdir(backup_dir)), sorted([*batches, self.REPORT_NAME])
        )
        with open(os.path.join(backup_dir, self.REPORT_NAME), encoding="utf-8") as handle:
            report = handle.read()
        self.assertIn(f"Original Directory:  {results_dir}\n", report)
        self.assertIn(f"Backup Directory:    {backup_dir}\n", report)
        reasons = report.split("REASON(S) FOR BACKUP:\n", 1)[1].split("\n\n", 1)[0]
        return [line.removeprefix("  - ") for line in reasons.splitlines()]
