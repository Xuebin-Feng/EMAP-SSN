"""The `reference` command resolves its target once and reports the row it uses."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager  # noqa: E402
from commands import reference as reference_command  # noqa: E402
from EMAPSSN_Viewer import MainViewer  # noqa: E402
from utilities.Localization import Message  # noqa: E402
from utilities.Sequence_Utils import reference_header_matches  # noqa: E402
from tests.sparse_alignment import load_manager, write_fasta  # noqa: E402


class ReferenceResolutionTests(unittest.TestCase):
    def run_reference(self, records, headers, *targets, configured="", offset=0, initial_reference=None):
        """Run `reference` for each target on a toy MSA.

        RECORDS are the MSA rows and HEADERS the network headers, so a network
        node can be absent from the MSA. CONFIGURED is the ALIGNMENT_REFERENCE
        setting and OFFSET the session's alignment offset. INITIAL_REFERENCE is
        the reference the session starts with, loaded as the Viewer loads it at
        startup, so it is inactive when the MSA lacks it; none by default. The
        Viewer's own load_global_alignment reloads the MSA. Returns the viewer,
        the terminal log, and the success messages reported to the command portal.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        msa_path = os.path.join(directory.name, "toy.fasta")
        write_fasta(msa_path, records)
        viewer = MainViewer.__new__(MainViewer)
        viewer.full_headers = list(headers)
        viewer.active_reference = initial_reference or ""
        viewer.alignment_offset = offset
        viewer.console_text = SimpleNamespace(text="")
        viewer.alignment = load_manager(msa_path, viewer.full_headers, reference=initial_reference)
        if initial_reference:
            viewer.alignment.set_offset(offset)
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
        # As the command portal records them: in English.
        messages = [str(call.args[1]) for call in succeeded.call_args_list]
        return viewer, output.getvalue(), messages

    def load_configured(self, records, headers, reference, setting=None):
        """Load a toy MSA as the Viewer does at startup, with ALIGNMENT_REFERENCE set.

        The Viewer, including an MCP launch, hands the setting to the alignment
        without running the `reference` command. SETTING is the
        ALIGNMENT_REFERENCE setting when it is not REFERENCE.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        msa_path = os.path.join(directory.name, "toy.fasta")
        write_fasta(msa_path, records)
        output = io.StringIO()
        with mock.patch.object(Alignment_Manager.cfg, "FILTER_MIN_OCCUPANCY", 50), \
                mock.patch.object(
                    Alignment_Manager.cfg, "ALIGNMENT_REFERENCE", reference if setting is None else setting
                ), \
                redirect_stdout(output):
            manager = Alignment_Manager.Alignment_Manager(
                msa_path, full_headers=list(headers), active_reference=reference
            )
        return manager, output.getvalue()

    def test_a_second_argument_is_refused_before_anything_changes(self):
        records = [("E1_RA", "MAKLCD"), ("S3", "MAKLCD")]
        viewer, _, _ = self.run_reference(records, [h for h, _ in records], "E1_RA")
        alignment = viewer.alignment
        engine = reference_command.Command_Engine

        with mock.patch.object(viewer, "load_global_alignment") as reload, \
                mock.patch.object(engine, "command_failed", wraps=engine.command_failed) as failed, \
                redirect_stdout(io.StringIO()):
            reference_command.run(viewer, ["S3", "WP_EXTRA", "E1_RA"])

        reload.assert_not_called()
        failed.assert_called_once()
        # The first argument after the target is the one named.
        self.assertIn("'WP_EXTRA' was not used", str(failed.call_args.args[1]))
        self.assertIn("was not used", viewer.console_text.text)
        self.assertEqual(viewer.active_reference, "E1_RA")
        self.assertIs(viewer.alignment, alignment)
        self.assertEqual(viewer.alignment.resolved_ref_full, "E1_RA")

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
        failed.assert_called_once()
        self.assertEqual((failed.call_args.args[0], str(failed.call_args.args[1])), (viewer, message))
        succeeded.assert_not_called()
        viewer.load_global_alignment.assert_not_called()
        self.assertIs(viewer.alignment, alignment)
        self.assertEqual(viewer.active_reference, "node1")
        self.assertEqual(viewer.console_text.text, message)
        self.assertEqual(output.getvalue(), f"\n{message}\n")

    def loaded_viewer_on_node1(self):
        """A viewer whose MSA is loaded with node1 as its reference."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        msa_path = os.path.join(directory.name, "toy.fasta")
        write_fasta(msa_path, [("node1", "MKC"), ("node2", "M-C")])
        viewer = MainViewer.__new__(MainViewer)
        viewer.full_headers = ["node1", "node2"]
        viewer.active_reference = "node1"
        viewer.console_text = SimpleNamespace(text="")
        viewer.alignment = load_manager(msa_path, viewer.full_headers, reference="node1")
        viewer.resolved_ref_full = "node1"
        return viewer

    def test_failed_reload_restores_the_previous_alignment_and_reference(self):
        viewer = self.loaded_viewer_on_node1()
        alignment = viewer.alignment

        def reload_returns_no_rows(reuse_loaded=False):
            # The loader ran but produced no alignment.
            viewer.alignment = SimpleNamespace(aln=None, has_reference=False)

        viewer.load_global_alignment = mock.Mock(side_effect=reload_returns_no_rows)
        engine = reference_command.Command_Engine
        with mock.patch.object(engine, "command_failed") as failed, \
                mock.patch.object(engine, "command_succeeded") as succeeded, \
                redirect_stdout(io.StringIO()):
            reference_command.run(viewer, ["node2"])

        viewer.load_global_alignment.assert_called_once_with(reuse_loaded=True)
        failed.assert_called_once()
        self.assertEqual(
            str(failed.call_args.args[1]),
            "Error: Could not reload the current MSA for reference 'node2'.",
        )
        succeeded.assert_not_called()
        self.assertIs(viewer.alignment, alignment)
        self.assertTrue(viewer.alignment.has_reference)
        self.assertEqual(viewer.active_reference, "node1")
        self.assertEqual(viewer.resolved_ref_full, "node1")

    def test_a_failed_reload_reports_the_loaders_reason(self):
        viewer = self.loaded_viewer_on_node1()
        alignment = viewer.alignment
        reason = Message("MSA rejected: {error}", error="MSA FASTA contains no records.")

        def reload_is_rejected(reuse_loaded=False):
            # The loader leaves its reason on the manager that the restore replaces.
            viewer.alignment = SimpleNamespace(aln=None, has_reference=False, load_failure=reason)

        viewer.load_global_alignment = mock.Mock(side_effect=reload_is_rejected)
        engine = reference_command.Command_Engine
        with mock.patch.object(engine, "command_failed") as failed,                 mock.patch.object(engine, "command_succeeded") as succeeded,                 redirect_stdout(io.StringIO()):
            reference_command.run(viewer, ["node2"])

        failed.assert_called_once()
        self.assertEqual(
            str(failed.call_args.args[1]),
            "Error: Could not reload the current MSA for reference 'node2'. "
            "MSA rejected: MSA FASTA contains no records.",
        )
        self.assertIn("MSA FASTA contains no records.", viewer.console_text.text)
        succeeded.assert_not_called()
        self.assertIs(viewer.alignment, alignment)
        self.assertEqual(viewer.active_reference, "node1")

    def test_reload_that_raises_restores_the_previous_state_and_reraises(self):
        viewer = self.loaded_viewer_on_node1()
        alignment = viewer.alignment
        viewer.load_global_alignment = mock.Mock(side_effect=RuntimeError("loader crashed"))

        with self.assertRaisesRegex(RuntimeError, "loader crashed"), \
                redirect_stdout(io.StringIO()):
            reference_command.run(viewer, ["node2"])

        self.assertIs(viewer.alignment, alignment)
        self.assertEqual(viewer.active_reference, "node1")
        self.assertEqual(viewer.resolved_ref_full, "node1")

    def test_bare_reference_marks_an_unresolved_reference_inactive(self):
        # The session starts with a reference the MSA lacks, as a configured
        # ALIGNMENT_REFERENCE can; the `reference` command cannot set one.
        viewer, _, _ = self.run_reference(
            [("node1", "AC")], ["node1", "node2"], "", initial_reference="node2"
        )

        self.assertFalse(viewer.alignment.has_reference)
        self.assertEqual(
            viewer.console_text.text,
            "Current Reference: node2 (inactive; not resolved in the current MSA)",
        )

    # The network has each target but the MSA lacks it. Another row's header
    # contains the target, or is contained in it; neither may anchor numbering.
    UNALIGNED_CASES = (
        (
            [("XE1_RA_variant", "MA--CD"), ("S3", "MAKLCD")],
            ["XE1_RA_variant", "E1_RA", "S3"],
            "E1_RA",
            "E1_RA",
        ),
        (
            [("P1", "MA--CD"), ("S3", "MAKLCD")],
            ["P1", "P12_kinase", "S3"],
            "P12_kinase",
            "P12_kinase",
        ),
        # A leading identifier is reported as the full header it resolves to.
        (
            [("P1", "MA--CD"), ("S3", "MAKLCD")],
            ["P1", "P12_kinase", "S3"],
            "P12",
            "P12_kinase",
        ),
    )

    def refuse(self, msa, network, target, **options):
        """Run `reference TARGET` over a working reference, S3, and an offset of 10.

        Returns the viewer, the terminal log, the mock of command_failed, the
        success messages reported, and the mocks of the Viewer's MSA reload and
        undo-state save.
        """
        engine = reference_command.Command_Engine
        with mock.patch.object(MainViewer, "load_global_alignment", autospec=True) as reload, \
                mock.patch.object(MainViewer, "_save_state", autospec=True) as save_state, \
                mock.patch.object(engine, "command_failed") as failed:
            viewer, log, messages = self.run_reference(
                msa, network, target, offset=10, initial_reference="S3", **options
            )
        return viewer, log, failed, messages, reload, save_state

    def test_reference_missing_from_the_msa_is_refused_and_changes_nothing(self):
        for msa, network, target, name in self.UNALIGNED_CASES:
            with self.subTest(target=target):
                viewer, log, failed, messages, reload, save_state = self.refuse(
                    msa, network, target
                )

                message = (
                    f"Error: '{name}' is not in the loaded alignment, so it cannot be "
                    "the reference. The reference is unchanged."
                )
                failed.assert_called_once()
                self.assertEqual(
                    (failed.call_args.args[0], str(failed.call_args.args[1])), (viewer, message)
                )
                self.assertEqual(messages, [])
                self.assertEqual(viewer.console_text.text, message)
                self.assertEqual(log, f"\n{message}\n")
                # The reference, its numbering and the offset are as they were,
                # the MSA was not reloaded, and there is no undo step.
                reload.assert_not_called()
                save_state.assert_not_called()
                self.assertEqual(viewer.active_reference, "S3")
                self.assertTrue(viewer.alignment.has_reference)
                self.assertEqual(viewer.alignment.resolved_ref_full, "S3")
                self.assertEqual(viewer.alignment_offset, 10)
                self.assertEqual(viewer.alignment.offset, 10)
                self.assertEqual(viewer.alignment.label_to_col["11"], 0)

    def test_wildcard_with_no_aligned_match_is_refused_by_the_pattern(self):
        msa = [("S3_other", "MAKLCD"), ("S4_other", "MAKLCD")]
        network = ["WP_0123.10_protein_B", "WP_0123.1_protein_A", "S3_other", "S4_other"]

        viewer, log, failed, messages, reload, save_state = self.refuse(msa, network, "WP_01*")

        message = (
            "Error: 'WP_01*' is not in the loaded alignment, so it cannot be the "
            "reference. The reference is unchanged."
        )
        failed.assert_called_once()
        self.assertEqual(str(failed.call_args.args[1]), message)
        self.assertEqual(messages, [])
        reload.assert_not_called()
        save_state.assert_not_called()
        self.assertEqual(viewer.active_reference, "S3")

    def test_wildcard_takes_the_first_aligned_match_over_an_earlier_unaligned_one(self):
        # WP_0123.10_protein_B comes first in the network but has no MSA row.
        msa = [("WP_0123.1_protein_A", "MAKLCD"), ("WP_0123.2_protein_C", "MAKLCD")]
        network = [
            "WP_0123.10_protein_B", "WP_0123.1_protein_A", "WP_0123.2_protein_C",
        ]

        viewer, log, messages = self.run_reference(msa, network, "WP_01*", offset=10)

        self.assertEqual(viewer.active_reference, "WP_0123.1_protein_A")
        self.assertTrue(viewer.alignment.has_reference)
        self.assertEqual(viewer.alignment.resolved_ref_full, "WP_0123.1_protein_A")
        self.assertEqual(viewer.alignment.offset, 10)
        self.assertEqual(messages, ["Reference successfully set: WP_0123.1_protein_A."])
        # The warning still names the header that is used.
        self.assertIn(
            "Multiple matches found for 'WP_01*'. Using 'WP_0123.1_protein_A'.", log
        )

    def test_wildcard_that_matches_one_aligned_header_gives_no_warning(self):
        msa = [("WP_0123.1_protein_A", "MAKLCD"), ("S3", "MAKLCD")]
        network = ["WP_0123.1_protein_A", "S3"]

        viewer, log, messages = self.run_reference(msa, network, "WP_01*")

        self.assertNotIn("Multiple matches", log)
        self.assertEqual(viewer.alignment.resolved_ref_full, "WP_0123.1_protein_A")

    def test_a_reload_that_leaves_the_reference_inactive_is_refused_too(self):
        # The header has a row in the MSA, so the reload should anchor on it.
        # Should it not, the previous reference stays instead of going inactive.
        viewer = self.loaded_viewer_on_node1()
        alignment = viewer.alignment

        def reload_without_the_reference(reuse_loaded=False):
            viewer.alignment = SimpleNamespace(aln=object(), has_reference=False)

        viewer.load_global_alignment = mock.Mock(side_effect=reload_without_the_reference)
        engine = reference_command.Command_Engine
        with mock.patch.object(engine, "command_failed") as failed, \
                mock.patch.object(engine, "command_succeeded") as succeeded, \
                redirect_stdout(io.StringIO()):
            reference_command.run(viewer, ["node2"])

        viewer.load_global_alignment.assert_called_once()
        failed.assert_called_once()
        self.assertEqual(
            str(failed.call_args.args[1]),
            "Error: 'node2' is not in the loaded alignment, so it cannot be the "
            "reference. The reference is unchanged.",
        )
        succeeded.assert_not_called()
        self.assertIs(viewer.alignment, alignment)
        self.assertEqual(viewer.active_reference, "node1")
        self.assertEqual(viewer.resolved_ref_full, "node1")

    def test_without_a_loaded_alignment_the_command_still_fails_on_the_reload(self):
        # No MSA is selected, so there is no sequence to check and the command
        # reloads as before; the reload has no alignment, and the command fails.
        engine = reference_command.Command_Engine
        with redirect_stdout(io.StringIO()):
            no_alignment = Alignment_Manager.Alignment_Manager("")
        for loaded in (no_alignment, None):
            with self.subTest(alignment=loaded):
                viewer = MainViewer.__new__(MainViewer)
                viewer.full_headers = ["node1", "node2"]
                viewer.active_reference = ""
                viewer.alignment_offset = 0
                viewer.console_text = SimpleNamespace(text="")
                viewer.alignment = loaded
                with mock.patch.object(Alignment_Manager.cfg, "MSA_FILE", ""), \
                        mock.patch.object(engine, "command_failed") as failed, \
                        mock.patch.object(engine, "command_succeeded") as succeeded, \
                        redirect_stdout(io.StringIO()):
                    reference_command.run(viewer, ["node1"])

                failed.assert_called_once()
                self.assertEqual(
                    str(failed.call_args.args[1]),
                    "Error: Could not reload the current MSA for reference 'node1'.",
                )
                succeeded.assert_not_called()
                self.assertIs(viewer.alignment, loaded)
                self.assertEqual(viewer.active_reference, "")

    def test_help_says_the_reference_must_be_in_the_alignment(self):
        viewer = SimpleNamespace(console_text=SimpleNamespace(text=""))
        output = io.StringIO()
        with redirect_stdout(output):
            reference_command.run(viewer, ["help"])

        text = " ".join(output.getvalue().split())
        self.assertIn("The reference must have a sequence in the loaded alignment.", text)
        self.assertIn("the first one that has a sequence in the loaded alignment is used", text)

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
        # When the configured reference is a sequence the MSA lacks, numbering
        # is pure occupancy: S3, the ALIGNMENT_REFERENCE setting, no longer
        # keeps its low-occupancy column 1.
        msa = [("alpha", "M-C"), ("beta", "M-C"), ("S3", "MKC")]

        manager, _ = self.load_configured(
            msa, ["alpha", "beta", "S3", "E1_RA"], "E1_RA", setting="S3"
        )

        self.assertFalse(manager.has_reference)
        self.assertEqual(manager.valid_cols, {0, 2})
        self.assertEqual(manager.label_to_col, {"1": 0, "2": 2})

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


class ReferenceReuseTests(unittest.TestCase):
    """`reference` renumbers the loaded rows instead of reading the MSA again."""

    RECORDS = [
        ("alpha", "MK-LC-D"),
        ("beta", "M--LCAD"),
        ("gamma", "-KQL--D"),
        ("delta", "MKQLCA-"),
    ]
    # epsilon is a network node the MSA lacks.
    HEADERS = ["alpha", "beta", "gamma", "delta", "epsilon"]

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.msa_path = os.path.join(self.directory, "toy.fasta")
        write_fasta(self.msa_path, self.RECORDS)
        for name, value in (("MSA_FILE", self.msa_path), ("FILTER_MIN_OCCUPANCY", 60)):
            patcher = mock.patch.object(Alignment_Manager.cfg, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def started_viewer(self, reference="alpha", offset=0):
        """A viewer that loaded the MSA at startup, as the Viewer does."""
        viewer = MainViewer.__new__(MainViewer)
        viewer.full_headers = list(self.HEADERS)
        viewer.active_reference = reference
        viewer.alignment_offset = offset
        viewer.console_text = SimpleNamespace(text="")
        with redirect_stdout(io.StringIO()):
            viewer.load_global_alignment()
        return viewer

    def run_reference(self, viewer, target):
        """Run `reference TARGET`, counting the MSA reads. Returns (reads, terminal log)."""
        output = io.StringIO()
        with mock.patch.object(
            Alignment_Manager, "load_alignment_with_reason", wraps=Alignment_Manager.load_alignment_with_reason
        ) as read, redirect_stdout(output):
            reference_command.run(viewer, [target])
        return read.call_count, output.getvalue()

    def assert_numbered_as_a_fresh_load(self, manager, reference, offset):
        with redirect_stdout(io.StringIO()):
            fresh = Alignment_Manager.Alignment_Manager(
                self.msa_path, full_headers=list(self.HEADERS),
                active_reference=reference, alignment_offset=offset,
            )
        for name in (
            "valid_cols", "col_to_label", "label_to_col", "resolved_ref_full", "has_reference",
            "offset", "matched_headers", "missing_headers", "seq_map", "_base_col_to_label",
        ):
            self.assertEqual(getattr(manager, name), getattr(fresh, name), name)
        for name in ("viewer_to_aln", "aligned_node_mask"):
            np.testing.assert_array_equal(getattr(manager, name), getattr(fresh, name), name)
        self.assertEqual(vars(manager.sanitization_stats), vars(fresh.sanitization_stats))
        self.assertEqual(manager.aln.headers, fresh.aln.headers)
        self.assertEqual((manager.aln.matrix != fresh.aln.matrix).nnz, 0)

    def test_a_new_reference_keeps_the_loaded_rows_and_numbers_them_afresh(self):
        viewer = self.started_viewer(offset=5)
        rows = viewer.alignment.aln

        for target in ("gamma", "beta", "alpha", "delta"):
            with self.subTest(reference=target):
                reads, log = self.run_reference(viewer, target)

                self.assertEqual(reads, 0)
                self.assertIs(viewer.alignment.aln, rows)
                self.assertIn("Reusing the loaded alignment; the file is unchanged", log)
                self.assertEqual(viewer.alignment.resolved_ref_full, target)
                self.assert_numbered_as_a_fresh_load(viewer.alignment, target, 5)

    def test_an_edited_file_is_read_again(self):
        viewer = self.started_viewer()
        rows = viewer.alignment.aln
        edited = [("alpha", "MKKLC-D"), *self.RECORDS[1:]]

        # The same size, so only the modification time tells the edit apart.
        write_fasta(self.msa_path, edited)
        status = os.stat(self.msa_path)
        os.utime(self.msa_path, ns=(status.st_atime_ns, status.st_mtime_ns + 2_000_000_000))
        reads, log = self.run_reference(viewer, "beta")

        self.assertEqual(reads, 1)
        self.assertIsNot(viewer.alignment.aln, rows)
        self.assertNotIn("Reusing", log)
        self.assertEqual(viewer.alignment.aln[0].seq, "MKKLC-D")
        self.assert_numbered_as_a_fresh_load(viewer.alignment, "beta", 0)

        # A longer file is read again too.
        write_fasta(self.msa_path, [(header, sequence + "W") for header, sequence in self.RECORDS])
        reads, _ = self.run_reference(viewer, "gamma")

        self.assertEqual(reads, 1)
        self.assertEqual(viewer.alignment.aln.get_alignment_length(), 8)

    def test_another_file_or_other_network_headers_are_read_afresh(self):
        with redirect_stdout(io.StringIO()):
            loaded = Alignment_Manager.Alignment_Manager(
                self.msa_path, full_headers=list(self.HEADERS), active_reference="alpha"
            )
        copy_path = os.path.join(self.directory, "copy.fasta")
        write_fasta(copy_path, self.RECORDS)
        os.utime(copy_path, ns=(os.stat(self.msa_path).st_atime_ns, os.stat(self.msa_path).st_mtime_ns))

        for path, headers in (
            (copy_path, self.HEADERS),
            (self.msa_path, self.HEADERS[:3]),
            (self.msa_path, list(reversed(self.HEADERS))),
        ):
            with self.subTest(path=os.path.basename(path), headers=headers), \
                    mock.patch.object(
                        Alignment_Manager, "load_alignment_with_reason",
                        wraps=Alignment_Manager.load_alignment_with_reason,
                    ) as read, redirect_stdout(io.StringIO()):
                manager = Alignment_Manager.Alignment_Manager(
                    path, full_headers=list(headers), active_reference="beta", reuse=loaded
                )
                self.assertEqual(read.call_count, 1)
                self.assertIsNot(manager.aln, loaded.aln)
                # The rows keep the MSA's order.
                self.assertEqual(manager.aln.headers, [h for h, _ in self.RECORDS if h in headers])

    def test_loading_without_reuse_always_reads_the_file(self):
        # `alignment` and the startup load read the file even when it is unchanged.
        viewer = self.started_viewer()
        rows = viewer.alignment.aln

        with mock.patch.object(
            Alignment_Manager, "load_alignment_with_reason", wraps=Alignment_Manager.load_alignment_with_reason
        ) as read, redirect_stdout(io.StringIO()):
            viewer.load_global_alignment()

        self.assertEqual(read.call_count, 1)
        self.assertIsNot(viewer.alignment.aln, rows)

    def test_a_deleted_file_is_not_reused_and_the_reference_is_kept(self):
        viewer = self.started_viewer()
        previous = viewer.alignment
        os.remove(self.msa_path)

        with mock.patch.object(reference_command.Command_Engine, "command_failed") as failed:
            reads, _ = self.run_reference(viewer, "beta")

        self.assertEqual(reads, 1)
        failed.assert_called_once()
        self.assertIs(viewer.alignment, previous)
        self.assertEqual(viewer.active_reference, "alpha")


if __name__ == "__main__":
    unittest.main()
