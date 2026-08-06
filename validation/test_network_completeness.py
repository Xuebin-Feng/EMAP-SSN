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

"""Tests for constant-time HDF5 network completeness inspection."""

import os
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Cache_Manifest import inspect_network_completeness


class NetworkCompletenessTests(unittest.TestCase):
    def _network(self, directory, name, sequence_count, edge_i, edge_j, score_name):
        path = os.path.join(directory, name)
        with h5py.File(path, "w") as network:
            network.create_dataset(
                "headers",
                data=np.asarray([f"seq_{index}" for index in range(sequence_count)], dtype="S"),
            )
            network.create_dataset("i", data=np.asarray(edge_i, dtype=np.uint32))
            network.create_dataset("j", data=np.asarray(edge_j, dtype=np.uint32))
            network.create_dataset(score_name, data=np.ones(len(edge_i), dtype=np.float32))
        return path

    def test_complete_embedding_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._network(
                directory,
                "embedding.h5",
                4,
                [0, 0, 0, 1, 1, 2],
                [1, 2, 3, 2, 3, 3],
                "g_score",
            )
            info = inspect_network_completeness(path)

        self.assertEqual(info.status, "complete")
        self.assertEqual(info.sequence_count, 4)
        self.assertEqual(info.edge_count, 6)
        self.assertEqual(info.expected_edge_count, 6)

    def test_incomplete_blast_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._network(
                directory,
                "blast.h5",
                4,
                [0, 0, 1],
                [1, 2, 3],
                "score",
            )
            info = inspect_network_completeness(path)

        self.assertEqual(info.status, "incomplete")
        self.assertEqual(info.edge_count, 3)
        self.assertEqual(info.expected_edge_count, 6)

    def test_mismatched_edge_shapes_are_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._network(directory, "bad.h5", 3, [0, 0], [1, 2], "score")
            with h5py.File(path, "a") as network:
                del network["j"]
                network.create_dataset("j", data=np.asarray([1], dtype=np.uint32))
            info = inspect_network_completeness(path)

        self.assertEqual(info.status, "unknown")
        self.assertIn("different lengths", info.reason)

    def test_missing_dataset_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "missing.h5")
            with h5py.File(path, "w") as network:
                network.create_dataset("headers", data=np.asarray([b"a", b"b"]))
                network.create_dataset("i", data=np.asarray([0], dtype=np.uint32))
            info = inspect_network_completeness(path)

        self.assertEqual(info.status, "unknown")
        self.assertIn("j", info.reason)

    def test_overfull_network_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._network(
                directory,
                "overfull.h5",
                3,
                [0, 0, 1, 1],
                [1, 2, 2, 0],
                "score",
            )
            info = inspect_network_completeness(path)

        self.assertEqual(info.status, "unknown")
        self.assertIn("exceeding", info.reason)

    def test_unreadable_path_is_unknown(self):
        info = inspect_network_completeness(os.path.join("missing", "network.h5"))
        self.assertEqual(info.status, "unknown")
        self.assertIn("Unable to inspect", info.reason)

    def test_probe_uses_metadata_without_reading_dataset_values(self):
        class ShapeOnlyDataset:
            def __init__(self, length):
                self.shape = (length,)
                self.ndim = 1

            def __getitem__(self, _key):
                raise AssertionError("dataset values must not be read")

            def __array__(self, *_args, **_kwargs):
                raise AssertionError("dataset values must not be materialized")

        class ShapeOnlyNetwork(dict):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        network = ShapeOnlyNetwork(
            headers=ShapeOnlyDataset(4),
            i=ShapeOnlyDataset(6),
            j=ShapeOnlyDataset(6),
        )
        with mock.patch.object(h5py, "File", return_value=network):
            info = inspect_network_completeness("shape-only.h5")

        self.assertEqual(info.status, "complete")


if __name__ == "__main__":
    unittest.main()
