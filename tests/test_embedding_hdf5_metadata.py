"""Metadata-first embedding databases and the tools that write or read them.

HDF5_Storage writes the manifest (headers, full sequences, model, saving mode)
before any embedding. Generate_Embeddings resumes against it, and
Embedding_Cropping, Embedding_Extraction, Embedding_Injection and
Embedding_SSEARCH read sequences from it instead of from a separate FASTA.
"""

import os
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import h5py
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
UTILITIES = SRC / "utilities"
TOOLS = SRC / "tools"
for path in (SRC, UTILITIES, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Embedding_Cropping
import Embedding_Extraction
from utilities import HDF5_Storage as Embedding_HDF5
import Embedding_Injection
import Embedding_SSEARCH
import Generate_Embeddings
from tools.tool_helpers import Remote_Requests


def create_complete_database(path, records, model_name="test_model", saving_mode="float16"):
    headers = [header for header, _ in records]
    sequences = [sequence for _, sequence in records]
    dtype = Embedding_HDF5.dtype_for_saving_mode(saving_mode)
    with h5py.File(path, "w") as hf:
        group = Embedding_HDF5.create_metadata_first_file(
            hf,
            headers,
            sequences,
            model_name,
            saving_mode,
        )
        for index, (header, sequence) in enumerate(records):
            group.create_dataset(
                header,
                data=np.full((len(sequence), 3), index + 1, dtype=dtype),
            )
        Embedding_HDF5.mark_generation_complete(hf)


class FakePlugin:
    SUPPORTED_MODELS = ["test_model"]
    MODEL_EXECUTION_MODES = {"test_model": "local"}

    def __init__(self):
        self.generated = []

    def load_model(self, model_name, device):
        return object()

    def get_embedding(self, sequence, model_obj, device, target_dtype):
        self.generated.append(sequence)
        return np.ones((len(sequence), 3), dtype=target_dtype)


class EmbeddingHdf5Tests(unittest.TestCase):
    def test_manifest_is_written_before_embedding_group(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "metadata_first.h5"
            with h5py.File(path, "w") as hf:
                Embedding_HDF5.write_embedding_manifest(
                    hf,
                    ["Alpha"],
                    ["ACDE"],
                    "test_model",
                    "float16",
                )
                self.assertIn("headers", hf)
                self.assertIn("sequences", hf)
                self.assertNotIn("embeddings", hf)
                self.assertFalse(bool(hf.attrs["generation_complete"]))
                self.assertNotIn("embedding_schema_version", hf.attrs)

    def test_complete_database_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "complete.h5"
            create_complete_database(path, [("Alpha", "ACDE"), ("Beta", "FG")])
            with h5py.File(path, "r") as hf:
                manifest = Embedding_HDF5.read_embedding_manifest(hf)
                self.assertEqual(manifest.headers, ["Alpha", "Beta"])
                self.assertEqual(manifest.sequences, ["ACDE", "FG"])
                self.assertEqual(manifest.saving_mode, "float16")
                self.assertTrue(manifest.generation_complete)
                self.assertEqual(manifest.feature_dimension, 3)

    def test_legacy_and_incomplete_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy = pathlib.Path(temp_dir) / "legacy.h5"
            with h5py.File(legacy, "w") as hf:
                hf.attrs["model_name"] = "test_model"
                hf.create_dataset("headers", data=[b"Alpha"])
                hf.create_group("embeddings")
            with h5py.File(legacy, "r") as hf:
                with self.assertRaisesRegex(ValueError, "Legacy embedding files"):
                    Embedding_HDF5.read_embedding_manifest(hf)

            incomplete = pathlib.Path(temp_dir) / "incomplete.h5"
            with h5py.File(incomplete, "w") as hf:
                Embedding_HDF5.create_metadata_first_file(
                    hf, ["Alpha"], ["ACDE"], "test_model", "float16"
                )
            with h5py.File(incomplete, "r") as hf:
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    Embedding_HDF5.read_embedding_manifest(hf)

    def test_unsafe_duplicate_or_malformed_records_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            Embedding_HDF5.validate_manifest_records(
                ["Alpha", "Alpha"], ["AA", "CC"]
            )
        with self.assertRaisesRegex(ValueError, "safe HDF5 key"):
            Embedding_HDF5.validate_manifest_records([".."], ["AA"])
        with self.assertRaisesRegex(ValueError, "path or null"):
            Embedding_HDF5.validate_manifest_records(["Alpha/Beta"], ["AA"])

        with tempfile.TemporaryDirectory() as temp_dir:
            malformed = pathlib.Path(temp_dir) / "malformed.h5"
            with h5py.File(malformed, "w") as hf:
                group = Embedding_HDF5.create_metadata_first_file(
                    hf, ["Alpha"], ["ACDE"], "test_model", "float16"
                )
                group.create_dataset("Alpha", data=np.ones((3, 2), dtype=np.float16))
            with h5py.File(malformed, "r") as hf:
                with self.assertRaisesRegex(ValueError, "4 residues"):
                    Embedding_HDF5.read_embedding_manifest(
                        hf, require_complete=False
                    )


class EmbeddingWriterTests(unittest.TestCase):
    def test_generate_resume_allows_reorder_and_addition(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            fasta = temp_path / "records.fasta"
            output = temp_path / "records.h5"
            fasta.write_text(">Alpha\nACDE\n>Beta\nFGH\n", encoding="utf-8")
            plugin = FakePlugin()

            with mock.patch.object(
                Generate_Embeddings,
                "DEVICE_SELECTION",
                "cpu",
            ):
                count = Generate_Embeddings.generate_embeddings(
                    fasta,
                    output,
                    "test_model",
                    "float16",
                    plugin_loader=lambda _: plugin,
                )
                self.assertEqual(count, 2)

                fasta.write_text(
                    ">Beta\nFGH\n>Alpha\nACDE\n>Gamma\nIK\n",
                    encoding="utf-8",
                )
                count = Generate_Embeddings.generate_embeddings(
                    fasta,
                    output,
                    "test_model",
                    "float16",
                    plugin_loader=lambda _: plugin,
                )
                self.assertEqual(count, 1)

            with h5py.File(output, "r") as hf:
                manifest = Embedding_HDF5.read_embedding_manifest(hf)
                self.assertEqual(manifest.headers, ["Beta", "Alpha", "Gamma"])
                self.assertEqual(set(hf["embeddings"]), set(manifest.headers))
            self.assertEqual(plugin.generated, ["ACDE", "FGH", "IK"])

    def test_generate_resumes_an_interrupted_partial_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            fasta = temp_path / "records.fasta"
            output = temp_path / "records.h5"
            fasta.write_text(">Alpha\nACDE\n>Beta\nFGH\n", encoding="utf-8")
            with h5py.File(output, "w") as hf:
                group = Embedding_HDF5.create_metadata_first_file(
                    hf,
                    ["Alpha", "Beta"],
                    ["ACDE", "FGH"],
                    "test_model",
                    "float16",
                )
                group.create_dataset(
                    "Alpha", data=np.ones((4, 3), dtype=np.float16)
                )
                hf.flush()

            plugin = FakePlugin()
            with mock.patch.object(
                Generate_Embeddings,
                "DEVICE_SELECTION",
                "cpu",
            ):
                count = Generate_Embeddings.generate_embeddings(
                    fasta,
                    output,
                    "test_model",
                    "float16",
                    plugin_loader=lambda _: plugin,
                )
            self.assertEqual(count, 1)
            self.assertEqual(plugin.generated, ["FGH"])
            with h5py.File(output, "r") as hf:
                self.assertTrue(
                    Embedding_HDF5.read_embedding_manifest(hf).generation_complete
                )

    def test_generate_rejects_removal_and_sequence_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            fasta = temp_path / "records.fasta"
            output = temp_path / "records.h5"
            fasta.write_text(">Alpha\nACDE\n>Beta\nFGH\n", encoding="utf-8")
            plugin = FakePlugin()
            with mock.patch.object(
                Generate_Embeddings,
                "DEVICE_SELECTION",
                "cpu",
            ):
                Generate_Embeddings.generate_embeddings(
                    fasta,
                    output,
                    "test_model",
                    "float16",
                    plugin_loader=lambda _: plugin,
                )

            fasta.write_text(">Alpha\nACDE\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "removed"):
                Generate_Embeddings.generate_embeddings(
                    fasta,
                    output,
                    "test_model",
                    "float16",
                    plugin_loader=lambda _: plugin,
                )
            fasta.write_text(">Alpha\nAAAA\n>Beta\nFGH\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                Generate_Embeddings.generate_embeddings(
                    fasta,
                    output,
                    "test_model",
                    "float16",
                    plugin_loader=lambda _: plugin,
                )

    def test_cropping_uses_stored_full_sequences(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            source = temp_path / "source.h5"
            cropped_fasta = temp_path / "cropped.fasta"
            output = temp_path / "cropped.h5"
            create_complete_database(source, [("Alpha_Beta", "AACDEFG")])
            cropped_fasta.write_text(">Alpha/Beta\ncde\n", encoding="utf-8")

            result = Embedding_Cropping.crop_embeddings(
                source, cropped_fasta, output
            )
            self.assertEqual(result["resolved_headers"], ["Alpha_Beta"])
            with h5py.File(output, "r") as hf:
                manifest = Embedding_HDF5.read_embedding_manifest(hf)
                self.assertEqual(manifest.sequences, ["CDE"])
                np.testing.assert_array_equal(
                    hf["embeddings"]["Alpha_Beta"][:],
                    np.ones((3, 3), dtype=np.float16),
                )

    def test_extraction_and_injection_reject_changed_sequences(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            source = temp_path / "source.h5"
            selection = temp_path / "selection.fasta"
            create_complete_database(source, [("Alpha", "ACDE")])
            selection.write_text(">Alpha\nAAAA\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "does not match"):
                Embedding_Extraction.extract_subset(
                    source, selection, temp_path / "extract.h5"
                )
            with self.assertRaisesRegex(ValueError, "changed"):
                Embedding_Injection.inject_embeddings(
                    source, selection, temp_path / "inject.h5"
                )

    def test_extraction_and_injection_preserve_parallel_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            source = temp_path / "source.h5"
            selection = temp_path / "selection.txt"
            replacement = temp_path / "replacement.fasta"
            extract_output = temp_path / "extract.h5"
            inject_output = temp_path / "inject.h5"
            create_complete_database(
                source, [("Alpha", "ACDE"), ("Beta", "FGH")]
            )
            selection.write_text("Beta\n", encoding="utf-8")
            Embedding_Extraction.extract_subset(
                source, selection, extract_output
            )
            with h5py.File(extract_output, "r") as hf:
                extracted = Embedding_HDF5.read_embedding_manifest(hf)
                self.assertEqual(extracted.headers, ["Beta"])
                self.assertEqual(extracted.sequences, ["FGH"])

            replacement.write_text(
                ">Beta\nFGH\n>Alpha\nACDE\n>Gamma\nIK\n",
                encoding="utf-8",
            )
            plugin = FakePlugin()
            with mock.patch.object(
                Embedding_Injection,
                "load_model",
                return_value=(object(), "cpu", plugin),
            ):
                _, generated, copied = Embedding_Injection.inject_embeddings(
                    source, replacement, inject_output
                )
            self.assertEqual((generated, copied), (1, 2))
            with h5py.File(inject_output, "r") as hf:
                injected = Embedding_HDF5.read_embedding_manifest(hf)
                self.assertEqual(injected.headers, ["Beta", "Alpha", "Gamma"])
                self.assertEqual(injected.sequences, ["FGH", "ACDE", "IK"])

    def test_ssearch_reads_database_metadata_without_fasta(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = pathlib.Path(temp_dir) / "source.h5"
            create_complete_database(source, [("Alpha", "ACDE")])
            with mock.patch.object(
                Embedding_SSEARCH, "FULL_INPUT_EMBED", str(source)
            ):
                manifest = Embedding_SSEARCH.prepare_database_embeddings()
            self.assertEqual(manifest.sequence_by_header, {"Alpha": "ACDE"})


class OrderedRemoteRunTests(unittest.TestCase):
    """Remote_Requests.run_in_order: several requests in flight, one writer."""

    def run_items(self, work, count=24, **options):
        consumed = []
        lines = []
        options.setdefault("rate_limit_wait", 0.001)
        Remote_Requests.run_in_order(
            list(range(count)),
            work,
            lambda index, item, result: consumed.append((index, item, result)),
            report=lines.append,
            **options,
        )
        return consumed, lines

    def test_results_arrive_in_input_order_whatever_finishes_first(self):
        # Earlier items take longer, so they finish last.
        def work(item):
            time.sleep(0.002 * (24 - item))
            return item * 10

        consumed, _ = self.run_items(work, max_in_flight=8)
        self.assertEqual(consumed, [(i, i, i * 10) for i in range(24)])

    def test_in_flight_and_unwritten_backlog_stay_bounded(self):
        lock = threading.Lock()
        state = {"running": 0, "peak": 0, "started": 0, "started_at_first_write": None}

        def work(item):
            with lock:
                state["running"] += 1
                state["started"] += 1
                state["peak"] = max(state["peak"], state["running"])
            try:
                time.sleep(0.3 if item == 0 else 0.001)
            finally:
                with lock:
                    state["running"] -= 1
            return item

        def consume(index, item, result):
            if state["started_at_first_write"] is None:
                state["started_at_first_write"] = state["started"]

        Remote_Requests.run_in_order(
            list(range(40)), work, consume, max_in_flight=4, report=lambda _: None
        )
        self.assertLessEqual(state["peak"], 4)
        # While item 0 was slow nothing was written, and no more than twice the
        # limit had been sent: the slow one plus a backlog of finished ones.
        self.assertLessEqual(state["started_at_first_write"], 8)

    def test_a_429_lowers_the_limit_and_the_sequence_is_sent_again(self):
        lock = threading.Lock()
        calls = {}
        state = {"running": 0, "peak_after_lowering": 0, "lowered": False}

        def work(item):
            with lock:
                state["running"] += 1
                running = state["running"]
                calls[item] = calls.get(item, 0) + 1
                first_call = calls[item] == 1
                if state["lowered"]:
                    state["peak_after_lowering"] = max(
                        state["peak_after_lowering"], running
                    )
            try:
                time.sleep(0.01)
                if item == 0 and first_call:
                    raise Remote_Requests.RateLimitedError("slow down")
                return item
            finally:
                with lock:
                    state["running"] -= 1

        consumed = []
        lines = []

        def report(text):
            lines.append(text)
            state["lowered"] = True

        Remote_Requests.run_in_order(
            list(range(12)),
            work,
            lambda index, item, result: consumed.append(result),
            max_in_flight=2,
            successes_per_increase=10**6,
            rate_limit_wait=0.001,
            report=report,
        )
        self.assertEqual(consumed, list(range(12)))
        self.assertEqual(calls[0], 2)
        self.assertEqual(len(lines), 1)
        self.assertIn("1 at a time", lines[0])
        self.assertEqual(state["peak_after_lowering"], 1)

    def test_successes_raise_the_limit_again_but_never_above_the_setting(self):
        lock = threading.Lock()
        seen = {"calls": 0, "running": 0, "peak": 0}

        def work(item):
            with lock:
                seen["calls"] += 1
                number = seen["calls"]
                seen["running"] += 1
                seen["peak"] = max(seen["peak"], seen["running"])
            try:
                time.sleep(0.005)
                if number <= 8:
                    # All eight first requests are turned away together.
                    raise Remote_Requests.RateLimitedError("slow down")
                return item
            finally:
                with lock:
                    seen["running"] -= 1

        consumed, lines = self.run_items(
            work, count=60, max_in_flight=8, successes_per_increase=3
        )
        self.assertEqual([item for _, item, _ in consumed], list(range(60)))
        # Eight answers at the same limit lower it once: 8 -> 4, not 8 -> 1.
        numbers = [
            int(line.split("sending ")[1].split(" at a time")[0]) for line in lines
        ]
        self.assertEqual(numbers[0], 4)
        self.assertEqual(max(numbers), 8)
        self.assertLessEqual(seen["peak"], 8)
        self.assertEqual(numbers[1:], sorted(numbers[1:]))

    def test_a_server_that_only_answers_429_ends_the_run_clearly(self):
        def work(item):
            raise Remote_Requests.RateLimitedError("slow down")

        consumed = []
        with self.assertRaisesRegex(RuntimeError, "too many requests"):
            Remote_Requests.run_in_order(
                [1, 2, 3],
                work,
                lambda *args: consumed.append(args),
                max_in_flight=2,
                max_resends=3,
                rate_limit_wait=0.001,
                report=lambda _: None,
            )
        self.assertEqual(consumed, [])

    def test_a_failure_keeps_what_was_in_order_and_ends_the_run(self):
        def work(item):
            if item == 5:
                time.sleep(0.05)
                raise RuntimeError("item five failed")
            time.sleep(0.001 if item < 5 else 0.5)
            return item

        consumed = []
        began = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "item five failed"):
            Remote_Requests.run_in_order(
                list(range(20)),
                work,
                lambda index, item, result: consumed.append(item),
                max_in_flight=6,
                report=lambda _: None,
            )
        self.assertEqual(consumed, [0, 1, 2, 3, 4])
        # The run did not wait for the slow requests still in flight.
        self.assertLess(time.monotonic() - began, 0.4)


class RemoteEmbeddingGenerationTests(unittest.TestCase):
    """Generate_Embeddings with a remote_api plugin and several requests at once."""

    class RemotePlugin:
        SUPPORTED_MODELS = ["remote_model"]
        MODEL_EXECUTION_MODES = {"remote_model": "remote_api"}

        def __init__(self, fail_on=None, rate_limit_every=0):
            self.fail_on = fail_on
            self.rate_limit_every = rate_limit_every
            self.lock = threading.Lock()
            self.calls = []
            self.running = 0
            self.peak = 0

        def load_model(self, model_name, device):
            return object()

        def get_embedding(self, sequence, model_obj, device, target_dtype):
            with self.lock:
                self.calls.append(sequence)
                number = len(self.calls)
                self.running += 1
                self.peak = max(self.peak, self.running)
            try:
                # Longer sequences take longer, so they finish out of order.
                time.sleep(0.0005 * len(sequence))
                if self.fail_on is not None and sequence == self.fail_on:
                    raise RuntimeError("stand-in server failure")
                if self.rate_limit_every and number % self.rate_limit_every == 0:
                    raise Remote_Requests.RateLimitedError("slow down")
                value = float(sum(map(ord, sequence)) % 97)
                return np.full((len(sequence), 3), value, dtype=target_dtype)
            finally:
                with self.lock:
                    self.running -= 1

    SEQUENCES = [
        "A" * (3 + (7 * index) % 11) + "CDEFGHIKLMNPQRSTVWY"[index % 19]
        for index in range(30)
    ]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.fasta = self.root / "records.fasta"
        self.fasta.write_text(
            "".join(f">S{i:02d}\n{seq}\n" for i, seq in enumerate(self.SEQUENCES)),
            encoding="utf-8",
        )

    def generate(self, plugin, output, parallel, saving_mode="float32"):
        with mock.patch.object(
            Generate_Embeddings, "API_PARALLEL_REQUESTS", parallel
        ), mock.patch.object(Remote_Requests, "RATE_LIMIT_WAIT_SECONDS", 0.001):
            return Generate_Embeddings.generate_embeddings(
                self.fasta,
                output,
                plugin.SUPPORTED_MODELS[0],
                saving_mode,
                plugin_loader=lambda _: plugin,
            )

    @staticmethod
    def read_all(path):
        with h5py.File(path, "r") as hf:
            data = {key: hf["embeddings"][key][()] for key in hf["embeddings"]}
            return data, bool(hf.attrs["generation_complete"])

    def test_concurrent_file_equals_serial_file_and_is_written_in_order(self):
        serial_plugin = self.RemotePlugin()
        serial = self.root / "serial.h5"
        self.generate(serial_plugin, serial, 1)
        self.assertEqual(serial_plugin.peak, 1)

        plugin = self.RemotePlugin()
        parallel = self.root / "parallel.h5"
        written = []
        real_validate = Generate_Embeddings.validate_embedding_array

        def recording_validate(embedding, sequence, *args, **kwargs):
            written.append(kwargs["header"])
            return real_validate(embedding, sequence, *args, **kwargs)

        with mock.patch.object(
            Generate_Embeddings, "validate_embedding_array", recording_validate
        ):
            count = self.generate(plugin, parallel, 8)
        self.assertEqual(count, len(self.SEQUENCES))
        self.assertEqual(written, [f"S{i:02d}" for i in range(len(self.SEQUENCES))])
        self.assertGreater(plugin.peak, 1)
        self.assertLessEqual(plugin.peak, 8)

        serial_data, serial_complete = self.read_all(serial)
        parallel_data, parallel_complete = self.read_all(parallel)
        self.assertTrue(serial_complete and parallel_complete)
        self.assertEqual(set(serial_data), set(parallel_data))
        for key, array in serial_data.items():
            np.testing.assert_array_equal(parallel_data[key], array)

    def test_429_answers_are_resent_and_the_file_is_still_complete(self):
        plugin = self.RemotePlugin(rate_limit_every=5)
        output = self.root / "limited.h5"
        self.generate(plugin, output, 6)
        data, complete = self.read_all(output)
        self.assertTrue(complete)
        self.assertEqual(len(data), len(self.SEQUENCES))
        self.assertGreater(len(plugin.calls), len(self.SEQUENCES))

    def test_a_failure_leaves_a_prefix_that_a_second_run_completes(self):
        failing = self.RemotePlugin(fail_on=self.SEQUENCES[12])
        output = self.root / "partial.h5"
        with self.assertRaisesRegex(RuntimeError, "stand-in server failure"):
            self.generate(failing, output, 4)
        data, complete = self.read_all(output)
        self.assertFalse(complete)
        headers = [f"S{i:02d}" for i in range(len(self.SEQUENCES))]
        # Only sequences before the failing one, written in order: a prefix.
        self.assertEqual(sorted(data), headers[: len(data)])
        self.assertLessEqual(len(data), 12)

        resumed = self.RemotePlugin()
        count = self.generate(resumed, output, 4)
        self.assertEqual(count, len(self.SEQUENCES) - len(data))
        self.assertEqual(
            sorted(resumed.calls),
            sorted(self.SEQUENCES[len(data):]),
        )
        data, complete = self.read_all(output)
        self.assertTrue(complete)
        self.assertEqual(len(data), len(self.SEQUENCES))

    def test_a_setting_of_one_keeps_the_one_at_a_time_path(self):
        plugin = self.RemotePlugin()
        with mock.patch.object(
            Generate_Embeddings, "run_in_order", side_effect=AssertionError
        ):
            self.generate(plugin, self.root / "one.h5", 1)
        self.assertEqual(plugin.peak, 1)

    def test_a_local_model_ignores_the_setting(self):
        plugin = FakePlugin()
        with mock.patch.object(
            Generate_Embeddings, "run_in_order", side_effect=AssertionError
        ), mock.patch.object(
            Generate_Embeddings, "API_PARALLEL_REQUESTS", 0
        ), mock.patch.object(Generate_Embeddings, "DEVICE_SELECTION", "cpu"):
            count = Generate_Embeddings.generate_embeddings(
                self.fasta,
                self.root / "local.h5",
                "test_model",
                "float16",
                plugin_loader=lambda _: plugin,
            )
        self.assertEqual(count, len(self.SEQUENCES))

    def test_invalid_values_are_refused_with_a_clear_message(self):
        for value in (0, 65, -3, "many", "8", 2.5, True, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "API_PARALLEL_REQUESTS must be a whole number from 1 to 64"
                ):
                    self.generate(self.RemotePlugin(), self.root / "bad.h5", value)
        self.assertFalse((self.root / "bad.h5").exists())
        for value in (1, 8, 64):
            self.assertEqual(
                Generate_Embeddings.validate_api_parallel_requests(value), value
            )

    def test_the_command_line_run_refuses_an_invalid_value(self):
        with mock.patch.object(
            Generate_Embeddings, "load_tool_settings"
        ), mock.patch.object(Generate_Embeddings, "API_PARALLEL_REQUESTS", 65):
            with self.assertRaises(SystemExit) as raised:
                Generate_Embeddings.main([])
        self.assertIn("API_PARALLEL_REQUESTS must be", str(raised.exception))


class EmbeddingCroppingTests(unittest.TestCase):
    """Crops copy the source rows at the crop's offset and report every skip."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        self.source = self.root / "source.h5"
        # Every row holds different values, so a wrong offset changes them.
        self.rows = {}
        records = [("Full", "MKACDEF"), ("Repeat", "AAGAA"), ("NoMatch", "WWW")]
        with h5py.File(self.source, "w") as hf:
            group = Embedding_HDF5.create_metadata_first_file(
                hf,
                [header for header, _ in records],
                [sequence for _, sequence in records],
                "test_model",
                "float16",
            )
            start = 0
            for header, sequence in records:
                size = 3 * len(sequence)
                data = np.arange(start, start + size, dtype=np.float16)
                self.rows[header] = data.reshape(len(sequence), 3)
                group.create_dataset(header, data=self.rows[header])
                start += size
            Embedding_HDF5.mark_generation_complete(hf)

    def write_crops(self, text):
        path = self.root / "cropped.fasta"
        path.write_bytes(text.encode("utf-8"))
        return path

    def test_crops_copy_rows_at_their_offset_and_skips_are_listed(self):
        cropped = self.write_crops(
            ">Full\nacd\n>Repeat\nAA\n>NoMatch\nYY\n>Absent\nMK\n"
        )
        output = self.root / "cropped.h5"

        result = Embedding_Cropping.crop_embeddings(self.source, cropped, output)

        self.assertEqual(result["output_hdf5"], output)
        self.assertEqual(result["resolved_headers"], ["Full", "Repeat"])
        # "AA" occurs twice in "AAGAA"; the first occurrence is used.
        self.assertEqual(result["ambiguous_resolved"], ["Repeat"])
        self.assertEqual(result["substring_not_found"], ["NoMatch"])
        self.assertEqual(result["missing_from_source"], ["Absent"])
        with h5py.File(output, "r") as hf:
            manifest = Embedding_HDF5.read_embedding_manifest(hf)
            self.assertEqual(manifest.headers, ["Full", "Repeat"])
            self.assertEqual(manifest.sequences, ["ACD", "AA"])
            self.assertEqual(manifest.saving_mode, "float16")
            self.assertEqual(manifest.model_name, "test_model")
            np.testing.assert_array_equal(
                hf["embeddings"]["Full"][:], self.rows["Full"][2:5]
            )
            np.testing.assert_array_equal(
                hf["embeddings"]["Repeat"][:], self.rows["Repeat"][0:2]
            )

    def test_output_that_resolves_to_the_source_is_rejected(self):
        cropped = self.write_crops(">Full\nACD\n")
        original = self.source.read_bytes()
        same_file = os.path.join(str(self.root), ".", "source.h5")

        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            Embedding_Cropping.crop_embeddings(self.source, cropped, same_file)

        self.assertEqual(self.source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
