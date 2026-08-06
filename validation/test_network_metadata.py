# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import tempfile
import unittest

import h5py
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Cache_Manifest import (
    NetworkMetadataError,
    build_canonical_cache_name,
    detect_network_type,
    read_network_metadata,
    validate_network_schema,
)


BLAST_DATASETS = ("headers", "i", "j", "score")
ALIGNMENT_DATASETS = (
    "headers",
    "seq_lens",
    "i",
    "j",
    "l_score",
    "l_len",
    "g_score",
    "g_len",
)


class NetworkMetadataTests(unittest.TestCase):
    def _network(self, directory, name, model_name, datasets):
        path = os.path.join(directory, name)
        with h5py.File(path, "w") as network:
            if model_name is not None:
                network.attrs["model_name"] = model_name
            for dataset in datasets:
                if dataset == "headers":
                    values = np.asarray(["seq_0", "seq_1"], dtype="S")
                elif dataset == "seq_lens":
                    values = np.asarray([10, 12], dtype=np.uint16)
                elif dataset in {"i", "j"}:
                    values = np.asarray([0], dtype=np.uint32)
                else:
                    values = np.asarray([1.0], dtype=np.float32)
                network.create_dataset(dataset, data=values)
        return path

    def test_blast_model_name_is_normalized_from_text_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, model_name in enumerate(("BLAST", " blast ", "BlAsT", b"BLAST")):
                path = self._network(
                    directory,
                    f"network_{index}.h5",
                    model_name,
                    BLAST_DATASETS,
                )
                metadata = read_network_metadata(path)
                self.assertEqual(metadata.network_type, "blast")
                self.assertEqual(metadata.model_name.casefold(), "blast")
                self.assertEqual(validate_network_schema(path), metadata)

    def test_any_other_nonempty_model_name_is_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, model_name in enumerate(("esmc_6b", "E1_RA", " custom model ")):
                path = self._network(
                    directory,
                    f"network_{index}.h5",
                    model_name,
                    ALIGNMENT_DATASETS,
                )
                metadata = validate_network_schema(path)
                self.assertEqual(metadata.network_type, "alignment")
                self.assertEqual(metadata.model_name, model_name.strip())

    def test_missing_blank_nontext_and_invalid_utf8_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = self._network(
                directory,
                "missing.h5",
                None,
                BLAST_DATASETS,
            )
            blank = self._network(
                directory,
                "blank.h5",
                "   ",
                BLAST_DATASETS,
            )
            numeric = self._network(
                directory,
                "numeric.h5",
                42,
                BLAST_DATASETS,
            )
            invalid_utf8 = self._network(
                directory,
                "invalid_utf8.h5",
                np.bytes_(b"\xff"),
                BLAST_DATASETS,
            )

            for path in (missing, blank, numeric, invalid_utf8):
                with self.subTest(path=os.path.basename(path)):
                    with self.assertRaisesRegex(NetworkMetadataError, "model_name"):
                        read_network_metadata(path)

    def test_filename_and_datasets_do_not_override_alignment_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            misleading_name = self._network(
                directory,
                "renamed_[BLAST]_EValue.h5",
                "esmc_6b",
                ALIGNMENT_DATASETS,
            )
            four_dataset_score_file = self._network(
                directory,
                "four_objects.h5",
                "E1_RA",
                BLAST_DATASETS,
            )

            self.assertEqual(detect_network_type(misleading_name), "alignment")
            self.assertEqual(detect_network_type(four_dataset_score_file), "alignment")
            with self.assertRaisesRegex(NetworkMetadataError, "seq_lens"):
                validate_network_schema(four_dataset_score_file)

    def test_schema_disagreement_fails_without_reclassification(self):
        with tempfile.TemporaryDirectory() as directory:
            blast_with_alignment_schema = self._network(
                directory,
                "blast_mismatch.h5",
                "BLAST",
                ALIGNMENT_DATASETS,
            )
            alignment_with_blast_schema = self._network(
                directory,
                "alignment_mismatch.h5",
                "esmc_6b",
                BLAST_DATASETS,
            )

            self.assertEqual(detect_network_type(blast_with_alignment_schema), "blast")
            with self.assertRaisesRegex(NetworkMetadataError, "score"):
                validate_network_schema(blast_with_alignment_schema)

            self.assertEqual(detect_network_type(alignment_with_blast_schema), "alignment")
            with self.assertRaisesRegex(NetworkMetadataError, "seq_lens"):
                validate_network_schema(alignment_with_blast_schema)

    def test_path_and_open_file_calls_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._network(
                directory,
                "network.h5",
                "BLAST",
                BLAST_DATASETS,
            )
            from_path = read_network_metadata(path)
            with h5py.File(path, "r") as network:
                self.assertEqual(read_network_metadata(network), from_path)
                self.assertEqual(validate_network_schema(network), from_path)
                self.assertEqual(detect_network_type(network), "blast")

    def test_cache_name_uses_metadata_instead_of_network_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            alignment_path = self._network(
                directory,
                "misleading_[BLAST]_EValue.h5",
                "esmc_6b",
                ALIGNMENT_DATASETS,
            )
            cache_name = build_canonical_cache_name(
                os.path.join(directory, "sequences.fasta"),
                alignment_path,
                "alignment",
                alignment_score="global",
                normalization="alignment_length",
                similarity_threshold=1.0,
            )
            self.assertEqual(
                cache_name,
                "sequences_[esmc_6b]_alignment_length_global_Score1.0",
            )


if __name__ == "__main__":
    unittest.main()
