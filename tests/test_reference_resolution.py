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
from EMAPSSN_Viewer import MainViewer  # noqa: E402
from utilities.Sequence_Utils import reference_header_matches  # noqa: E402
from tests.sparse_alignment import load_manager, write_fasta  # noqa: E402


class ReferenceResolutionTests(unittest.TestCase):
    def run_reference(self, records, headers, *targets, configured="", offset=0):
        """Run `reference` for each target on a toy MSA.

        RECORDS are the MSA rows and HEADERS the network headers, so a network
        node can be absent from the MSA. CONFIGURED is the ALIGNMENT_REFERENCE
        setting and OFFSET the session's alignment offset. The Viewer's own
        load_global_alignment reloads the MSA. Returns the viewer, the terminal
        log, and the success messages reported to the command portal.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        msa_path = os.path.join(directory.name, "toy.fasta")
        write_fasta(msa_path, records)
        viewer = MainViewer.__new__(MainViewer)
        viewer.full_headers = list(headers)
        viewer.active_reference = ""
        viewer.alignment_offset = offset
        viewer.console_text = SimpleNamespace(text="")
        viewer.alignment = load_manager(msa_path, viewer.full_headers)
        output = io.StringIO()
        engine = reference_command.Command_Engine
        with mock.patch.object(Alignment_Manager.cfg, "MSA_FILE", msa_path), \
                mock.patch.object(Alignment_Manager.cfg, "FILTER_MIN_OCCUPANCY", 50), \
                mock.patch.object(Alignment_Manager.cfg, "ALIGNMENT_REFERENCE", configured), \
                mock.patch.object(
                    engine, "command_succeeded", wraps=engine.command_succeeded
                ) as succeeded, \
                redirect_stdout(output):
            for target in targets:
                reference_command.run(viewer, [target] if target else [])
        messages = [call.args[1] for call in succeeded.call_args_list]
        return viewer, output.getvalue(), messages

    def load_configured(self, records, headers, reference):
        """Load a toy MSA as the Viewer does at startup, with ALIGNMENT_REFERENCE set.

        The Viewer, including an MCP launch, hands the setting to the alignment
        without running the `reference` command.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        msa_path = os.path.join(directory.name, "toy.fasta")
        write_fasta(msa_path, records)
        output = io.StringIO()
        with mock.patch.object(Alignment_Manager.cfg, "FILTER_MIN_OCCUPANCY", 50), \
                mock.patch.object(Alignment_Manager.cfg, "ALIGNMENT_REFERENCE", reference), \
                redirect_stdout(output):
            manager = Alignment_Manager.Alignment_Manager(
                msa_path, full_headers=list(headers), active_reference=reference
            )
        return manager, output.getvalue()

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

    def test_exact_header_wins_over_a_longer_header_that_starts_with_it(self):
        # `E1_RA` is also the leading identifier of the earlier `E1_RA_variant`.
        records = [("E1_RA_variant", "MA--CD"), ("E1_RA", "MAKLCD")]

        viewer, log, messages = self.run_reference(
            records, [h for h, _ in records], "E1_RA"
        )

        self.assertNotIn("Multiple matches", log)
        self.assertEqual(viewer.alignment.resolved_ref_full, "E1_RA")
        self.assertEqual(messages, ["Reference successfully set: E1_RA."])

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

    def test_matches_list_every_tie_at_the_winning_tier(self):
        # The Config GUI's consistency check reports these, so it can name the
        # header the Viewer will use and the others that tie with it.
        headers = [h for h, _ in self.ACCESSION_RECORDS]

        self.assertEqual(
            reference_header_matches(headers, "WP_0123.1"), ["WP_0123.1_protein_A"]
        )
        self.assertEqual(
            reference_header_matches(headers, "WP_01*"),
            ["WP_0123.10_protein_B", "WP_0123.1_protein_A"],
        )
        self.assertEqual(reference_header_matches(headers, "absent"), [])

    def test_wildcard_can_select_a_version_suffix(self):
        records = self.ACCESSION_RECORDS

        viewer, log, _ = self.run_reference(records, [h for h, _ in records], "*.1")

        self.assertNotIn("Multiple matches", log)
        self.assertEqual(viewer.alignment.resolved_ref_full, "WP_0123.1_protein_A")

    def test_unknown_target_fails_without_reloading(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        msa_path = os.path.join(directory.name, "toy.fasta")
        write_fasta(msa_path, [("node1", "MKC"), ("node2", "M-C")])
        viewer = MainViewer.__new__(MainViewer)
        viewer.full_headers = ["node1", "node2"]
        viewer.active_reference = "node1"
        viewer.console_text = SimpleNamespace(text="")
        alignment = load_manager(msa_path, viewer.full_headers, reference="node1")
        viewer.alignment = alignment
        viewer.load_global_alignment = mock.Mock()
        engine = reference_command.Command_Engine
        with mock.patch.object(engine, "command_failed") as failed, \
                mock.patch.object(engine, "command_succeeded") as succeeded, \
                redirect_stdout(io.StringIO()) as output:
            reference_command.run(viewer, ["zzz"])

        message = "Error: Reference 'zzz' not found."
        failed.assert_called_once_with(viewer, message)
        succeeded.assert_not_called()
        viewer.load_global_alignment.assert_not_called()
        self.assertIs(viewer.alignment, alignment)
        self.assertEqual(viewer.active_reference, "node1")
        self.assertEqual(viewer.console_text.text, message)
        self.assertEqual(output.getvalue(), f"\n{message}\n")

    def test_bare_reference_marks_an_unresolved_reference_inactive(self):
        viewer, _, _ = self.run_reference(
            [("node1", "AC")], ["node1", "node2"], "node2", ""
        )

        self.assertFalse(viewer.alignment.has_reference)
        self.assertEqual(
            viewer.console_text.text,
            "Current Reference: node2 (inactive; not resolved in the current MSA)",
        )

    def test_reference_missing_from_the_msa_stays_inactive(self):
        # The network has the node but the MSA lacks it. Another row's header
        # contains the target, or is contained in it; neither may anchor numbering.
        cases = (
            (
                [("XE1_RA_variant", "MA--CD"), ("S3", "MAKLCD")],
                ["XE1_RA_variant", "E1_RA", "S3"],
                "E1_RA",
            ),
            (
                [("P1", "MA--CD"), ("S3", "MAKLCD")],
                ["P1", "P12_kinase", "S3"],
                "P12_kinase",
            ),
        )
        for msa, network, target in cases:
            with self.subTest(target=target):
                viewer, _, messages = self.run_reference(msa, network, target)

                self.assertFalse(viewer.alignment.has_reference)
                self.assertEqual(viewer.alignment.resolved_ref_full, "None")
                self.assertEqual(
                    messages,
                    [
                        f"Reference '{target}' is configured but inactive because it "
                        "is not present in the current MSA. Pure occupancy mode "
                        "remains active."
                    ],
                )
                self.assertIn("configured but inactive", viewer.console_text.text)

    def test_reference_reload_keeps_the_session_offset(self):
        # `offset 10` earlier in the session. Switching the reference reloads
        # the MSA, which must apply the same offset to the new anchor.
        viewer, _, messages = self.run_reference(
            [("node1", "MKC"), ("node2", "M-C")],
            ["node1", "node2"],
            "node2",
            "node1",
            offset=10,
        )

        self.assertEqual(
            messages,
            ["Reference successfully set: node2.", "Reference successfully set: node1."],
        )
        self.assertEqual(viewer.alignment.resolved_ref_full, "node1")
        self.assertEqual(viewer.alignment_offset, 10)
        self.assertEqual(viewer.alignment.offset, 10)
        self.assertEqual(viewer.alignment.label_to_col, {"11": 0, "12": 1, "13": 2})

    def test_occupancy_mode_keeps_no_columns_for_the_configured_reference(self):
        # After `reference` selects a sequence the MSA lacks, numbering is pure
        # occupancy: S3, the ALIGNMENT_REFERENCE setting, no longer keeps its
        # low-occupancy column 1.
        msa = [("alpha", "M-C"), ("beta", "M-C"), ("S3", "MKC")]

        viewer, _, _ = self.run_reference(
            msa, ["alpha", "beta", "S3", "E1_RA"], "E1_RA", configured="S3"
        )

        self.assertFalse(viewer.alignment.has_reference)
        self.assertEqual(viewer.alignment.valid_cols, {0, 2})
        self.assertEqual(viewer.alignment.label_to_col, {"1": 0, "2": 2})

    def test_configured_reference_resolves_like_the_command(self):
        manager, log = self.load_configured(
            self.ACCESSION_RECORDS,
            [h for h, _ in self.ACCESSION_RECORDS],
            "WP_0123.1",
        )

        self.assertNotIn("Multiple matches", log)
        self.assertEqual(manager.resolved_ref_full, "WP_0123.1_protein_A")

        records = [("WP1_E1_RA_protein", "MAC-D"), ("WP2_other_protein", "MACKD")]
        manager, _ = self.load_configured(records, [h for h, _ in records], "WP1*")

        self.assertTrue(manager.has_reference)
        self.assertEqual(manager.resolved_ref_full, "WP1_E1_RA_protein")

    def test_configured_reference_missing_from_the_msa_stays_inactive(self):
        msa = [("XE1_RA_variant", "MA--CD"), ("S3", "MAKLCD")]

        manager, log = self.load_configured(
            msa, ["XE1_RA_variant", "E1_RA", "S3"], "E1_RA"
        )

        self.assertFalse(manager.has_reference)
        self.assertEqual(manager.resolved_ref_full, "None")
        self.assertIn("Configured reference 'E1_RA' is missing", log)


if __name__ == "__main__":
    unittest.main()
