import os
import pathlib
import sys
import unittest
from unittest import mock


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILITIES_DIR = PROJECT_ROOT / "src" / "utilities"
TOOLS_DIR = PROJECT_ROOT / "src" / "tools"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import Embedding_MSA


class EmbeddingMsaConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.fasta_dir = os.path.join("project", "Input_Files", "Sequence_Sets")
        self.embed_dir = os.path.join("project", "Embeddings")
        self.network_dir = os.path.join(
            "project", "Input_Files", "Networks_EValues"
        )
        self.msa_dir = os.path.join("project", "Input_Files", "Multiple_Alignments")
        self.input_embed = "uniprotkb_IPR011343_90_[esmc_6b]_embeddings.h5"
        self.input_network = "uniprotkb_IPR011343_90_[esmc_6b]_network.h5"

    def resolve(self, *, input_fasta="", input_embed=None, input_network=None, use_filter=False):
        return Embedding_MSA.resolve_msa_configuration(
            self.fasta_dir,
            self.embed_dir,
            self.network_dir,
            input_fasta,
            self.input_embed if input_embed is None else input_embed,
            self.input_network if input_network is None else input_network,
            use_filter,
        )

    def test_empty_fasta_is_not_joined_when_filter_is_disabled(self):
        original_join = os.path.join
        with mock.patch.object(
            Embedding_MSA.os.path, "join", wraps=original_join
        ) as join_mock:
            resolved = self.resolve()

        self.assertEqual(resolved["full_input_fasta"], "")
        self.assertNotIn(
            (self.fasta_dir, ""),
            [call.args for call in join_mock.call_args_list],
        )
        self.assertEqual(resolved["sequence_set"], "uniprotkb_IPR011343_90")

    def test_current_gui_settings_produce_expected_output_name(self):
        resolved = self.resolve(input_fasta="", use_filter=False)
        output_path = Embedding_MSA.build_msa_output_path(
            self.msa_dir,
            resolved["sequence_set"],
            "esmc_6b",
        )

        self.assertEqual(
            os.path.basename(output_path),
            "uniprotkb_IPR011343_90_[esmc_6b]_alignment.fasta",
        )

    def test_enabled_filter_requires_a_selected_fasta(self):
        with self.assertRaisesRegex(
            Embedding_MSA.MSAConfigurationError,
            "INPUT_FASTA must select a FASTA file",
        ):
            self.resolve(input_fasta="", use_filter=True)

    def test_enabled_filter_uses_selected_fasta_name_and_path(self):
        resolved = self.resolve(
            input_fasta="Selected_subset.fasta",
            use_filter=True,
        )

        self.assertEqual(
            resolved["full_input_fasta"],
            os.path.join(self.fasta_dir, "Selected_subset.fasta"),
        )
        self.assertEqual(resolved["sequence_set"], "Selected_subset")

    def test_required_hdf5_settings_have_field_specific_errors(self):
        cases = (
            ("", self.input_network, "INPUT_EMBED"),
            (self.input_embed, "", "INPUT_NETWORK"),
        )
        for input_embed, input_network, expected_field in cases:
            with self.subTest(expected_field=expected_field):
                with self.assertRaisesRegex(
                    Embedding_MSA.MSAConfigurationError,
                    expected_field,
                ):
                    self.resolve(
                        input_embed=input_embed,
                        input_network=input_network,
                    )

    def test_noncanonical_embedding_name_uses_stem_fallback(self):
        resolved = self.resolve(input_embed="custom_embeddings.h5")
        self.assertEqual(resolved["sequence_set"], "custom")


if __name__ == "__main__":
    unittest.main()
