import pathlib
import sys
import tempfile
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
import Embedding_HDF5
import Embedding_Injection
import Embedding_SSEARCH
import Generate_Embeddings


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
            self.assertFalse(hasattr(Embedding_Cropping, "FULL_FASTA"))
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
            self.assertFalse(hasattr(Embedding_SSEARCH, "INPUT_FASTA"))
            self.assertFalse(hasattr(Embedding_SSEARCH, "FULL_INPUT_FASTA"))


if __name__ == "__main__":
    unittest.main()
