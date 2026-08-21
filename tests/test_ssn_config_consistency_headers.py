import pathlib
import sys
import tempfile
import unittest

import h5py


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import SSN_Config  # noqa: E402


class ConsistencyHeaderSanitizationTests(unittest.TestCase):
    def test_fasta_headers_use_canonical_sanitization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = pathlib.Path(temp_dir) / "sequences.fasta"
            fasta_path.write_text(
                ">  Alpha   Beta  \nac-d\n",
                encoding="utf-8",
            )

            headers = SSN_Config._load_consistency_fasta_headers(fasta_path)

        self.assertEqual(headers, ["Alpha_Beta"])

    def test_fasta_and_sparse_msa_headers_share_the_canonical_rule(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            msa_fasta_path = temp_path / "alignment.fasta"
            msa_hdf5_path = temp_path / "alignment.h5"
            msa_fasta_path.write_text(
                ">Alpha   Beta\nAC-D\n",
                encoding="utf-8",
            )
            with h5py.File(msa_hdf5_path, "w") as hf:
                hf.create_dataset("headers", data=[b"Alpha   Beta"])

            fasta_headers = SSN_Config._load_consistency_msa_headers(
                msa_fasta_path
            )
            hdf5_headers = SSN_Config._load_consistency_msa_headers(
                msa_hdf5_path
            )

        self.assertEqual(fasta_headers, ["Alpha_Beta"])
        self.assertEqual(hdf5_headers, ["Alpha_Beta"])


if __name__ == "__main__":
    unittest.main()
