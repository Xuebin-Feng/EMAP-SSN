"""The `reference` command resolves its target once and reports the row it uses."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager  # noqa: E402
from commands import reference as reference_command  # noqa: E402
from tests.test_incomplete_alignment_commands import load_manager, write_fasta  # noqa: E402


class ReferenceResolutionTests(unittest.TestCase):
    def run_reference(self, records, headers, *targets):
        """Run `reference` for each target on a toy MSA.

        Returns the viewer, the terminal log, and the success messages reported
        to the command portal.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        msa_path = os.path.join(directory.name, "toy.fasta")
        write_fasta(msa_path, records)
        viewer = SimpleNamespace(
            full_headers=list(headers),
            active_reference="",
            alignment_offset=0,
            console_text=SimpleNamespace(text=""),
        )
        viewer.alignment = load_manager(msa_path, viewer.full_headers)

        def load_global_alignment():
            viewer.alignment = Alignment_Manager.Alignment_Manager(
                msa_path,
                full_headers=viewer.full_headers,
                active_reference=viewer.active_reference,
            )

        viewer.load_global_alignment = load_global_alignment
        output = io.StringIO()
        engine = reference_command.Command_Engine
        with mock.patch.object(Alignment_Manager.cfg, "FILTER_MIN_OCCUPANCY", 50), \
                mock.patch.object(Alignment_Manager.cfg, "ALIGNMENT_REFERENCE", ""), \
                mock.patch.object(
                    engine, "command_succeeded", wraps=engine.command_succeeded
                ) as succeeded, \
                redirect_stdout(output):
            for target in targets:
                reference_command.run(viewer, [target] if target else [])
        messages = [call.args[1] for call in succeeded.call_args_list]
        return viewer, output.getvalue(), messages

    def test_wildcard_target_resolves_and_activates_the_reference(self):
        records = [("WP1_E1_RA_protein", "MAC-D"), ("WP2_other_protein", "MACKD")]

        viewer, _, _ = self.run_reference(records, [h for h, _ in records], "WP1*")

        self.assertTrue(viewer.alignment.has_reference)
        self.assertEqual(viewer.active_reference, "WP1_E1_RA_protein")
        self.assertEqual(viewer.alignment.resolved_ref_full, "WP1_E1_RA_protein")
        self.assertEqual(viewer.console_text.text, "Reference successfully set.")

    def test_exact_header_wins_over_an_earlier_substring_match(self):
        # Network order lists a longer header containing the target first.
        records = [("XE1_RA_variant", "MA--CD"), ("E1_RA", "MAKLCD"), ("S3", "MAKLCD")]

        viewer, log, messages = self.run_reference(
            records, [h for h, _ in records], "E1_RA", ""
        )

        self.assertNotIn("Multiple matches", log)
        self.assertEqual(viewer.active_reference, "E1_RA")
        self.assertEqual(viewer.alignment.resolved_ref_full, "E1_RA")
        self.assertEqual(
            messages, ["Reference successfully set: E1_RA.", "Current Reference: E1_RA"]
        )
        # Numbering is anchored on E1_RA, which has a residue in every column.
        self.assertEqual(viewer.alignment.label_to_col["5"], 4)

    def test_ambiguous_target_reports_the_row_the_alignment_uses(self):
        records = [("XE1_RA_variant", "MA--CD"), ("E1_RA", "MAKLCD"), ("S3", "MAKLCD")]

        viewer, log, messages = self.run_reference(
            records, [h for h, _ in records], "E1_R", ""
        )

        # The warning, the success report, and the numbering anchor now agree.
        self.assertIn("Multiple matches found for 'E1_R'. Using 'XE1_RA_variant'.", log)
        self.assertEqual(viewer.alignment.resolved_ref_full, "XE1_RA_variant")
        self.assertEqual(
            messages,
            [
                "Reference successfully set: XE1_RA_variant.",
                "Current Reference: XE1_RA_variant",
            ],
        )

    # Canonical headers replace whitespace with `_`, so an accession is the
    # header's leading segment and `WP_0123.1` is a substring of `WP_0123.10`.
    ACCESSION_RECORDS = [
        ("WP_0123.10_protein_B", "MA--CD"),
        ("WP_0123.1_protein_A", "MAKLCD"),
        ("S3_other", "MAKLCD"),
    ]

    def test_versioned_accession_resolves_to_its_own_sequence(self):
        records = self.ACCESSION_RECORDS

        viewer, log, messages = self.run_reference(
            records, [h for h, _ in records], "WP_0123.1"
        )

        self.assertNotIn("Multiple matches", log)
        self.assertEqual(viewer.active_reference, "WP_0123.1_protein_A")
        self.assertEqual(viewer.alignment.resolved_ref_full, "WP_0123.1_protein_A")
        self.assertEqual(messages, ["Reference successfully set: WP_0123.1_protein_A."])

    def test_wildcard_can_select_a_version_suffix(self):
        records = self.ACCESSION_RECORDS

        viewer, log, _ = self.run_reference(records, [h for h, _ in records], "*.1")

        self.assertNotIn("Multiple matches", log)
        self.assertEqual(viewer.alignment.resolved_ref_full, "WP_0123.1_protein_A")

    def test_bare_reference_marks_an_unresolved_reference_inactive(self):
        viewer, _, _ = self.run_reference(
            [("node1", "AC")], ["node1", "node2"], "node2", ""
        )

        self.assertFalse(viewer.alignment.has_reference)
        self.assertEqual(
            viewer.console_text.text,
            "Current Reference: node2 (inactive; not resolved in the current MSA)",
        )


if __name__ == "__main__":
    unittest.main()
