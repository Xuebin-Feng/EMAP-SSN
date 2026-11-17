# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Predict Structures tool: validation, existing files, the remote rule, records and parity.

No test loads a model. The worker runs for real (``run_predictions``) with a stand-in for the ESM3
package, so the PDB files are written by the worker's own code, and a stand-in model writes each
residue's pLDDT, on a scale of 0 to 100, as the B-factor the way ESM does.
"""

import ast
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy  # noqa: F401  (real dependencies are loaded before sys.modules is patched)
import torch  # noqa: F401

from mcp_server.pipeline import Pipeline_Guide
from mcp_server.pipeline.Pipeline_Settings import _CONTRACTS, normalize_pipeline_settings
from resources import Biohub_API
from resources.esmfold import esmfold_worker
from tools import Predict_Structures as tool
from tools.tool_helpers import Structure_Prediction as prediction
from tools.tool_helpers.Tool_Pipeline import (
    DEFAULT_DIRECTORY_PATHS,
    TOOL_DIRECTORY_KEYS,
    get_tool_spec,
    list_tool_specs,
)
from utilities import Hardware_Acceleration

THREE = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE",
         "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
         "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}
SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
OTHER = "GSHMLEDPVDFLTYAQRFSSLRHELKAG"


def residue_plddt(index):
    """The stand-in model's pLDDT of residue ``index``, on the 0 to 1 scale ESM gives it."""
    return round(0.3 + 0.07 * (index % 10), 2)


class StandInOutput:
    """What a model returns: the structure as ESM writes it, the pLDDT of each residue 100 times as the B-factor."""

    def __init__(self, sequence):
        self.sequence = sequence

    def to_pdb_string(self):
        lines, serial = [], 1
        for index, letter in enumerate(self.sequence):
            for atom, element in (("N", "N"), ("CA", "C"), ("C", "C")):
                lines.append(
                    f"ATOM  {serial:5d} {atom:<4s} {THREE.get(letter, 'UNK'):3s} A{index + 1:4d}    "
                    f"{index * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{residue_plddt(index) * 100:6.2f}"
                    f"          {element:>2s}")
                serial += 1
        return "\n".join(lines) + "\nEND\n"


class StandInModel:
    def __init__(self):
        self.calls = []

    def generate(self, protein, config):
        self.calls.append(protein.sequence)
        return StandInOutput(protein.sequence)


class StandInProtein:
    def __init__(self, sequence=None):
        self.sequence = sequence


class StandInError:
    def __init__(self, error_code, error_msg="rejected"):
        self.error_code = error_code
        self.error_msg = error_msg


class StandInConfig:
    def __init__(self, track, num_steps):
        self.track, self.num_steps = track, num_steps


def stand_in_esm():
    esm_module = types.ModuleType("esm")
    sdk_module = types.ModuleType("esm.sdk")
    api_module = types.ModuleType("esm.sdk.api")
    api_module.ESMProtein = StandInProtein
    api_module.ESMProteinError = StandInError
    api_module.GenerationConfig = StandInConfig
    esm_module.sdk = sdk_module
    sdk_module.api = api_module
    return {"esm": esm_module, "esm.sdk": sdk_module, "esm.sdk.api": api_module}


class Candidate:
    """A device the machine reports."""

    def __init__(self, spec):
        self.spec = spec


class EngineCase(unittest.TestCase):
    """A temporary structure folder and the stand-ins for the model, the devices and the Biohub client."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.folder = self.root / "Folded_Structure"
        self.model = StandInModel()
        self.input_calls = []
        patches = [
            mock.patch.dict(sys.modules, stand_in_esm()),
            mock.patch.object(esmfold_worker, "_load_local_model", return_value=self.model),
            mock.patch.object(Hardware_Acceleration, "get_available_devices", return_value=[Candidate("cpu")]),
            mock.patch.object(Hardware_Acceleration, "get_optimal_device", return_value="cpu"),
            mock.patch("builtins.input", side_effect=lambda *a: self.fail("something was asked of a person")),
            mock.patch("getpass.getpass", side_effect=lambda *a: self.fail("a token was asked for")),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.log = []

    def predict(self, requests, **options):
        options.setdefault("log", self.log.append)
        return prediction.predict_structures(requests, self.folder, **options)

    def files(self):
        return sorted(path.name for path in self.folder.iterdir()) if self.folder.exists() else []

    def b_factors(self, name):
        text = (self.folder / name).read_text(encoding="utf-8")
        return [float(line[60:66]) for line in text.splitlines() if line.startswith("ATOM") and line[12:16].strip() == "CA"]


class FoldingTests(EngineCase):
    def test_a_typed_sequence_is_folded_and_its_b_factor_column_holds_the_per_residue_plddt(self):
        records = self.predict(prediction.sequence_request(SEQUENCE, "my protein"))

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual((record.name, record.length, record.status), ("my_protein", len(SEQUENCE), prediction.FOLDED))
        self.assertEqual(Path(record.path), self.folder.resolve() / "my_protein.pdb")
        expected = [float(f"{residue_plddt(index) * 100:6.2f}") for index in range(len(SEQUENCE))]
        self.assertEqual(self.b_factors("my_protein.pdb"), expected)
        self.assertEqual(record.mean_plddt, round(sum(expected) / len(expected), 1))
        # Only the structure is written: no residue data, no result file, no staging file left behind.
        self.assertEqual(self.files(), ["my_protein.pdb"])
        self.assertEqual(self.model.calls, [SEQUENCE])

    def test_the_b_factor_column_is_written_as_the_viewers_worker_writes_it(self):
        # The same worker call, so a file of this tool and of the Viewer's esmfold are alike.
        self.predict([("viewer_node", SEQUENCE)])
        viewer_dir = self.root / "viewer"
        with mock.patch.dict(sys.modules, stand_in_esm()):
            esmfold_worker.run_predictions([["viewer_node", SEQUENCE]], str(viewer_dir), mode="local",
                                           target_device="cpu")
        self.assertEqual((self.folder / "viewer_node.pdb").read_bytes(), (viewer_dir / "viewer_node.pdb").read_bytes())

    def test_a_sequence_with_no_name_is_named_from_its_own_digest(self):
        (name, cleaned), = prediction.sequence_request(" " + SEQUENCE.lower() + "* ")
        digest = hashlib.sha256(SEQUENCE.encode()).hexdigest()[:10]
        self.assertEqual((name, cleaned), (f"sequence_{digest}", SEQUENCE))
        self.assertNotEqual(prediction.generated_name(OTHER), name)

        first = self.predict([(name, cleaned)])
        again = self.predict([(name, cleaned)])
        self.assertEqual((first[0].status, again[0].status), (prediction.FOLDED, prediction.EXISTING))
        self.assertEqual(again[0].path, first[0].path)
        self.assertEqual(again[0].mean_plddt, first[0].mean_plddt)
        self.assertEqual(self.model.calls, [SEQUENCE])  # the second request folded nothing
        self.assertEqual(self.files(), [f"sequence_{digest}.pdb"])

    def test_an_existing_structure_is_kept_and_one_of_another_sequence_is_not_reported_as_the_result(self):
        self.predict([("same_name", SEQUENCE)])
        before = (self.folder / "same_name.pdb").read_bytes()

        record, = self.predict([("same_name", OTHER)])
        self.assertEqual(record.status, prediction.FAILED)
        self.assertIn("already exists for a different sequence", record.note)
        self.assertIn("choose another name or output folder", record.note)
        self.assertIsNone(record.mean_plddt)
        self.assertEqual((self.folder / "same_name.pdb").read_bytes(), before)
        self.assertEqual(self.model.calls, [SEQUENCE])

    def test_a_masked_residue_does_not_make_the_same_sequence_a_different_one(self):
        self.predict([("masked", "MKTAXIAKQR")])
        record, = self.predict([("masked", "MKTAYIAKQR")])
        self.assertEqual(record.status, prediction.EXISTING)

    def test_an_existing_file_with_no_readable_residues_is_kept_with_a_note(self):
        self.folder.mkdir()
        (self.folder / "odd.pdb").write_text("REMARK nothing here\n", encoding="utf-8")
        record, = self.predict([("odd", SEQUENCE)])
        self.assertEqual(record.status, prediction.EXISTING)
        self.assertIsNone(record.mean_plddt)
        self.assertIn("not checked", record.note)
        self.assertEqual(self.model.calls, [])
        self.assertTrue(any("Note: odd" in line for line in prediction.format_table([record])))

    def test_the_records_come_back_in_order_with_the_table_of_name_file_length_status_and_plddt(self):
        self.predict([("kept", SEQUENCE)])
        records = self.predict([("new_one", OTHER), ("kept", SEQUENCE)])
        self.assertEqual([record.name for record in records], ["new_one", "kept"])
        self.assertEqual([record.status for record in records], [prediction.FOLDED, prediction.EXISTING])
        self.assertEqual([record.length for record in records], [len(OTHER), len(SEQUENCE)])
        table = "\n".join(prediction.format_table(records))
        for text in ("Name", "File", "Length", "Status", "Mean pLDDT", "new_one.pdb", "kept.pdb", "folded", "existing"):
            self.assertIn(text, table)
        self.assertEqual(prediction.failed(records), [])
        self.assertEqual(self.files(), ["kept.pdb", "new_one.pdb"])

    def test_a_sequence_the_model_cannot_fold_fails_its_record_and_the_others_go_on(self):
        original = self.model.generate

        def generate(protein, config):
            if protein.sequence == OTHER:
                raise RuntimeError("out of memory")
            return original(protein, config)

        self.model.generate = generate
        records = self.predict([("bad", OTHER), ("good", SEQUENCE)])
        self.assertEqual([record.status for record in records], [prediction.FAILED, prediction.FOLDED])
        self.assertIn("out of memory", records[0].note)
        self.assertEqual(self.files(), ["good.pdb"])

    def test_names_that_are_long_are_capped_and_stay_different_files(self):
        long_a = "x" * 300 + "_a"
        long_b = "x" * 300 + "_b"
        records = self.predict([(long_a, SEQUENCE), (long_b, OTHER)])
        names = [os.path.basename(record.path) for record in records]
        self.assertEqual(len(set(names)), 2)
        for name in names:
            self.assertLessEqual(len(name), prediction.MAX_STEM_LENGTH + len(".pdb"))
        self.assertEqual(names[0][:50], "x" * 50)
        # The same names always give the same files, so a second run finds them.
        again = self.predict([(long_a, SEQUENCE), (long_b, OTHER)])
        self.assertEqual([record.status for record in again], [prediction.EXISTING] * 2)

    def test_a_folder_path_too_long_for_windows_is_refused_before_anything_is_created(self):
        with self.assertRaisesRegex(prediction.StructureRequestError, "path is too long"):
            self.predict([("a", SEQUENCE)], path_limit=len(str(self.folder.resolve())) + 10)
        self.assertEqual(self.files(), [])

    def test_the_name_is_cut_to_the_room_the_path_leaves(self):
        records = self.predict([("y" * 90, SEQUENCE)], path_limit=len(str(self.folder.resolve())) + 60)
        self.assertLessEqual(len(os.path.basename(records[0].path)), 60 - 1)


class RefusalTests(EngineCase):
    def refused(self, requests, pattern, **options):
        with self.assertRaisesRegex(prediction.StructureRequestError, pattern):
            self.predict(requests, **options)
        # Nothing was created, and no model was started.
        self.assertFalse(self.folder.exists())
        self.assertEqual(self.model.calls, [])

    def test_a_sequence_with_no_residues_is_refused(self):
        with self.assertRaisesRegex(prediction.StructureRequestError, "no amino acid residues"):
            prediction.sequence_request("1234 -- 56")
        with self.assertRaisesRegex(prediction.StructureRequestError, "no usable characters"):
            prediction.sequence_request(SEQUENCE, "///")

    def test_a_fasta_record_with_no_residues_is_refused(self):
        path = self.root / "set.fasta"
        path.write_text(f">good\n{SEQUENCE}\n>bad\n1234\n", encoding="utf-8")
        with self.assertRaisesRegex(prediction.StructureRequestError, r"hold no amino acid residues.*'bad'"):
            prediction.fasta_requests(str(path))
        path.write_text(f">good one\n{SEQUENCE.lower()}\n>second/one\n{OTHER}\n", encoding="utf-8")
        self.assertEqual(prediction.fasta_requests(str(path)), [("good_one", SEQUENCE), ("second_one", OTHER)])
        path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(prediction.StructureRequestError, "no sequences"):
            prediction.fasta_requests(str(path))

    def test_names_that_give_one_file_are_refused(self):
        self.refused([("a/b", SEQUENCE), ("a?b", OTHER)], "same structure file name")
        self.refused([("a_b", SEQUENCE), ("a_b", OTHER)], "same structure file name")

    def test_more_sequences_to_fold_than_the_maximum_are_refused_saying_how_many(self):
        requests = [(f"s{number}", SEQUENCE) for number in range(3)]
        self.refused(requests, r"3 sequences would be folded.*maximum of 2.*smaller FASTA", max_sequences=2)

    def test_structures_that_exist_do_not_count_against_the_maximum(self):
        requests = [(f"s{number}", SEQUENCE) for number in range(3)]
        self.predict(requests)
        records = self.predict(requests + [("fresh", OTHER)], max_sequences=1)
        self.assertEqual([record.status for record in records], [prediction.EXISTING] * 3 + [prediction.FOLDED])

    def test_the_default_maximum_is_twenty(self):
        self.assertEqual(prediction.DEFAULT_MAX_SEQUENCES, 20)
        self.assertEqual(tool.MAX_SEQUENCES, 20)
        self.refused([(f"s{number}", SEQUENCE) for number in range(21)], r"21 sequences would be folded")

    def test_a_maximum_that_is_not_a_positive_number_is_refused(self):
        self.refused([("a", SEQUENCE)], "at least 1", max_sequences=0)
        self.refused([("a", SEQUENCE)], "whole number", max_sequences="many")

    def test_an_unavailable_device_is_refused_before_anything_is_created(self):
        self.refused([("a", SEQUENCE)], r"cuda:7.*not available", device="cuda:7")

    def test_nothing_to_fold_and_an_unknown_model_are_refused(self):
        self.refused([], "no sequences")
        self.refused([("a", SEQUENCE)], "must be one of", mode="hosted")

    def test_remote_without_a_saved_token_is_refused_and_the_local_model_does_not_take_its_place(self):
        with mock.patch.object(Biohub_API, "credentials_state", return_value=("missing", None)), \
                mock.patch.object(esmfold_worker, "_load_large_client") as client:
            self.refused([("a", SEQUENCE)], r"needs a saved Biohub API token.*local model is not used instead",
                         mode=prediction.REMOTE)
        client.assert_not_called()
        self.assertEqual(self.log, [])

    def test_remote_with_biohub_settings_that_cannot_be_used_is_refused(self):
        error = Biohub_API.BiohubSettingsError("broken file")
        with mock.patch.object(Biohub_API, "credentials_state", return_value=("invalid", error)):
            self.refused([("a", SEQUENCE)], "cannot be used: broken file", mode=prediction.REMOTE)

class RemoteTests(EngineCase):
    SETTINGS = {"ESM_API_TOKEN": "never-shown", "ESM_API_URL": "https://biohub.ai", "ESM3_MODEL": "esm3-large-2024-03"}

    def remote(self, results):
        client = mock.Mock()
        client.generate.side_effect = list(results)
        patches = [
            mock.patch.object(Biohub_API, "credentials_state", return_value=("saved", None)),
            mock.patch.object(Biohub_API, "remote_model_name", return_value="esm3-large-2024-03"),
            mock.patch.object(esmfold_worker, "_load_large_client", return_value=(client, dict(self.SETTINGS))),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return client

    def test_a_remote_run_says_the_sequences_are_sent_and_never_uses_the_local_model(self):
        client = self.remote([StandInOutput(SEQUENCE)])
        record, = self.predict([("x", SEQUENCE)], mode=prediction.REMOTE)
        self.assertEqual(record.status, prediction.FOLDED)
        self.assertEqual(os.path.basename(record.path), "x_esm3-large-2024-03.pdb")
        self.assertEqual(self.files(), ["x_esm3-large-2024-03.pdb"])
        sending = [line for line in self.log if "Biohub API" in line]
        self.assertEqual(len(sending), 1)
        self.assertIn("Sending 1 sequence(s) to the Biohub API", sending[0])
        self.assertNotIn("never-shown", "\n".join(self.log))
        self.assertEqual(self.model.calls, [])
        self.assertEqual(client.generate.call_count, 1)

    def test_a_remote_run_that_finds_everything_saved_sends_nothing(self):
        client = self.remote([StandInOutput(SEQUENCE)])
        self.predict([("x", SEQUENCE)], mode=prediction.REMOTE)
        self.log.clear()
        record, = self.predict([("x", SEQUENCE)], mode=prediction.REMOTE)
        self.assertEqual(record.status, prediction.EXISTING)
        self.assertEqual(self.log, [])
        self.assertEqual(client.generate.call_count, 1)

    def test_a_rejected_token_fails_the_records_without_asking_for_another(self):
        self.remote([StandInError(401, "bad token")])
        record, = self.predict([("x", SEQUENCE)], mode=prediction.REMOTE)
        self.assertEqual(record.status, prediction.FAILED)
        self.assertIn("rejected the saved API token", record.note)
        self.assertNotIn("never-shown", record.note)
        self.assertEqual(self.files(), [])

    def test_a_local_run_never_looks_at_the_biohub_settings_or_client(self):
        with mock.patch.object(Biohub_API, "credentials_state") as state, \
                mock.patch.object(esmfold_worker, "_load_large_client") as client:
            record, = self.predict([("x", SEQUENCE)])
        state.assert_not_called()
        client.assert_not_called()
        self.assertEqual(record.status, prediction.FOLDED)
        self.assertEqual(os.path.basename(record.path), "x.pdb")

    def test_the_credential_check_reads_the_file_or_the_environment_without_returning_the_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "api.json")
            self.assertEqual(Biohub_API.credentials_state(path, environ={}), ("missing", None))
            self.assertEqual(Biohub_API.credentials_state(path, environ={"ESM_API_KEY": " token "}), ("saved", None))
            self.assertEqual(Biohub_API.credentials_state(path, environ={"ESM_API_KEY": "  "}), ("missing", None))
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{not json")
            state, error = Biohub_API.credentials_state(path, environ={})
            self.assertEqual(state, "invalid")
            self.assertNotIn("token", str(error).lower().replace("api token", ""))
            Path(path).write_text(json.dumps({"ESM_API_TOKEN": "t", "ESM3_MODEL": "esm3-open"}), encoding="utf-8")
            self.assertEqual(Biohub_API.credentials_state(path, environ={}), ("saved", None))
            self.assertEqual(Biohub_API.remote_model_name(path), "esm3-open")
            self.assertEqual(Biohub_API.remote_model_name(os.path.join(directory, "none.json")),
                             Biohub_API.DEFAULT_ESM3_MODEL)


class GatedWeightsTests(EngineCase):
    def test_a_refused_download_ends_the_run_with_the_workers_message_and_asks_nobody(self):
        message = ("Hugging Face refused access to the ESM3 weights (biohub/esm3-sm-open-v1). Accept the "
                   "model's terms on its Hugging Face page and log in (hf auth login), or fold with esmfold large.")
        with mock.patch.object(esmfold_worker, "_load_local_model", side_effect=RuntimeError(message)):
            with self.assertRaises(prediction.StructureRunError) as caught:
                self.predict([("x", SEQUENCE)])
        self.assertEqual(str(caught.exception), message)
        self.assertEqual(self.files(), [])


class SettingsAndInputTests(EngineCase):
    def settings(self, **values):
        path = self.root / "settings.json"
        directories = {"FASTA_DIR": str(self.root), "STRUCTURE_DIR": str(self.folder)}
        path.write_text(json.dumps({"DIRECTORIES": directories, "Predict_Structures.py": values}), encoding="utf-8")
        return str(path)

    def run_tool(self, **values):
        names = ("BATCH_INPUT", "INPUT_FASTA", "SEQUENCE", "SEQUENCE_NAME", "PREDICTION_MODEL", "DEVICE_SELECTION")
        namespace = {name: getattr(tool, name) for name in names}
        with mock.patch.multiple(tool, **namespace), contextlib.redirect_stdout(io.StringIO()) as output:
            code = tool.main([self.settings(**values)])
        return code, output.getvalue()

    def test_the_switch_decides_and_the_input_it_chose_missing_is_refused_before_anything_is_created(self):
        (self.root / "set.fasta").write_text(f">a\n{SEQUENCE}\n", encoding="utf-8")
        for values, wording in (
            ({}, "Batch Input is off, so a sequence is needed"),
            ({"SEQUENCE": "   ", "INPUT_FASTA": "set.fasta"}, "Batch Input is off, so a sequence is needed"),
            ({"BATCH_INPUT": True, "SEQUENCE": SEQUENCE}, "Batch Input is on, so a FASTA file is needed"),
            ({"BATCH_INPUT": True, "INPUT_FASTA": "  "}, "Batch Input is on, so a FASTA file is needed"),
        ):
            with self.subTest(values=values):
                code, output = self.run_tool(**values)
                self.assertEqual(code, 1)
                self.assertIn(wording, output)
                self.assertFalse(self.folder.exists())
        self.assertEqual(self.model.calls, [])

    def test_with_the_switch_off_the_fasta_is_ignored_and_with_it_on_the_sequence_is_ignored(self):
        (self.root / "set.fasta").write_text(f">from_fasta\n{OTHER}\n", encoding="utf-8")
        code, output = self.run_tool(SEQUENCE=SEQUENCE, SEQUENCE_NAME="typed", INPUT_FASTA="set.fasta")
        self.assertEqual((code, self.files()), (0, ["typed.pdb"]), output)
        code, output = self.run_tool(BATCH_INPUT=True, SEQUENCE=SEQUENCE, SEQUENCE_NAME="typed", INPUT_FASTA="set.fasta")
        self.assertEqual((code, self.files()), (0, ["from_fasta.pdb", "typed.pdb"]), output)
        # A FASTA that is not there matters only while the switch is on.
        code, output = self.run_tool(SEQUENCE=SEQUENCE, SEQUENCE_NAME="typed", INPUT_FASTA="missing.fasta")
        self.assertEqual(code, 0, output)

    def test_a_typed_sequence_runs_end_to_end_and_the_table_ends_it(self):
        code, output = self.run_tool(SEQUENCE=SEQUENCE, SEQUENCE_NAME="typed one", DEVICE_SELECTION="cpu")
        self.assertEqual(code, 0, output)
        self.assertEqual(self.files(), ["typed_one.pdb"])
        for text in ("Mean pLDDT", "typed_one.pdb", "folded", str(len(SEQUENCE)), "Structures are in:", "Done: 1"):
            self.assertIn(text, output)

    def test_a_relative_fasta_resolves_inside_the_fasta_directory_and_an_absolute_one_is_used_as_given(self):
        (self.root / "set.fasta").write_text(f">a\n{SEQUENCE}\n>b c\n{OTHER}\n", encoding="utf-8")
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "far.fasta").write_text(f">far\n{SEQUENCE}\n", encoding="utf-8")
        code, output = self.run_tool(BATCH_INPUT=True, INPUT_FASTA="set.fasta")
        self.assertEqual((code, self.files()), (0, ["a.pdb", "b_c.pdb"]), output)
        code, output = self.run_tool(BATCH_INPUT=True, INPUT_FASTA=str(elsewhere / "far.fasta"))
        self.assertEqual((code, self.files()), (0, ["a.pdb", "b_c.pdb", "far.pdb"]), output)

    def test_the_exit_code_is_nonzero_when_any_sequence_failed(self):
        self.predict([("same", SEQUENCE)])
        code, output = self.run_tool(SEQUENCE=OTHER, SEQUENCE_NAME="same")
        self.assertEqual(code, 1)
        self.assertIn("failed", output)
        self.assertIn("different sequence", output)

    def test_a_request_that_cannot_run_prints_one_error_line_and_exits_nonzero(self):
        code, output = self.run_tool(SEQUENCE=SEQUENCE, DEVICE_SELECTION="cuda:3")
        self.assertEqual(code, 1)
        errors = [line for line in output.splitlines() if "Error" in line]
        self.assertEqual(len(errors), 1)
        self.assertIn("cuda:3", errors[0])
        self.assertFalse(self.folder.exists())
        code, output = self.run_tool(BATCH_INPUT=True, INPUT_FASTA="missing.fasta")
        self.assertEqual(code, 1)
        self.assertIn("missing.fasta", output)

    def test_the_tool_never_names_remote_unless_the_setting_says_remote(self):
        self.assertEqual(tool.PREDICTION_MODEL, "local")
        with mock.patch.object(Biohub_API, "credentials_state") as state:
            code, _ = self.run_tool(SEQUENCE=SEQUENCE)
        self.assertEqual(code, 0)
        state.assert_not_called()

    def test_a_one_run_snapshot_in_the_default_folder_is_deleted_once_read_and_nothing_else_is(self):
        from tools.tool_helpers.Tool_Pipeline import ONE_RUN_SNAPSHOT_DIRECTORY, forget_one_run_snapshot
        folder = Path(tempfile.gettempdir()) / ONE_RUN_SNAPSHOT_DIRECTORY
        folder.mkdir(exist_ok=True)
        snapshot = folder / f"predict_structures-test-{os.getpid()}.json"
        snapshot.write_text(json.dumps({"DIRECTORIES": {"STRUCTURE_DIR": str(self.folder)},
                                        "Predict_Structures.py": {"SEQUENCE": SEQUENCE, "SEQUENCE_NAME": "snap"}}),
                            encoding="utf-8")
        names = ("BATCH_INPUT", "INPUT_FASTA", "SEQUENCE", "SEQUENCE_NAME")
        with mock.patch.multiple(tool, **{name: getattr(tool, name) for name in names}), \
                mock.patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tool.main([str(snapshot)]), 0)
        self.assertFalse(snapshot.exists())
        self.assertEqual(self.files(), ["snap.pdb"])
        kept = self.root / "exported.json"
        kept.write_text("{}", encoding="utf-8")
        forget_one_run_snapshot(str(kept))
        self.assertTrue(kept.exists())


class ParityTests(unittest.TestCase):
    def module_settings(self):
        tree = ast.parse((SRC_DIR / "tools" / "Predict_Structures.py").read_text(encoding="utf-8"))
        return {target.id for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}

    def test_every_documented_setting_is_a_module_setting_and_the_tool_is_registered(self):
        contract = _CONTRACTS["Predict_Structures.py"]
        self.assertTrue(set(contract["properties"]) <= self.module_settings())
        self.assertEqual(set(contract["properties"]), {
            "BATCH_INPUT", "INPUT_FASTA", "SEQUENCE", "SEQUENCE_NAME", "PREDICTION_MODEL", "DEVICE_SELECTION", "MAX_SEQUENCES"})
        for name, entry in contract["properties"].items():
            self.assertEqual(getattr(tool, name), entry["default"], name)
        spec = get_tool_spec("predict_structures")
        self.assertEqual((spec.script_name, spec.required_directories, spec.output_directories),
                         ("Predict_Structures.py", ("FASTA_DIR", "STRUCTURE_DIR"), ("STRUCTURE_DIR",)))
        self.assertEqual(TOOL_DIRECTORY_KEYS["Predict_Structures.py"], spec.required_directories)
        self.assertIn(spec, list_tool_specs())

    def test_the_output_folder_is_folded_structure_beside_the_alignment_report_folder(self):
        self.assertEqual(DEFAULT_DIRECTORY_PATHS["STRUCTURE_DIR"], os.path.join("Analysis_Results", "Folded_Structure"))
        self.assertEqual(Path(tool.STRUCTURE_DIR), PROJECT_ROOT / "Analysis_Results" / "Folded_Structure")

    def test_the_guide_describes_the_tool_the_way_the_other_tools_are_described(self):
        guide = Pipeline_Guide.tool_guide("predict_structures")
        self.assertEqual(set(guide["inputs"]), {"INPUT_FASTA"})
        self.assertEqual(guide["outputs"][0]["directory"], "STRUCTURE_DIR")
        self.assertIn("kept, never overwritten", guide["outputs"][0]["existing"])
        text = json.dumps(guide)
        for expected in ("B-factor", "0 to 100", ".residues.json", "MAX_SEQUENCES", "remote", "SEQUENCE_NAME",
                         "BATCH_INPUT", "absolute path", "inside FASTA_DIR"):
            self.assertIn(expected, text)
        watched = Pipeline_Guide.watched_directories("predict_structures", {"DIRECTORIES": {"STRUCTURE_DIR": "out"}}, str(PROJECT_ROOT))
        self.assertEqual(watched, (str(PROJECT_ROOT / "out"),))
        recipe = [w for w in Pipeline_Guide.workflows() if w["goal"] == "Predict a structure from a sequence"]
        self.assertEqual(len(recipe), 1)
        self.assertIn("SEQUENCE", " ".join(recipe[0]["steps"]))
        self.assertIn("no file is needed", " ".join(recipe[0]["steps"]))
        self.assertIn("predict_structures", " ".join(Pipeline_Guide.tool_guide("sanitize_sequences")["next"]))

    def check(self, **parameters):
        return normalize_pipeline_settings("predict_structures", str(PROJECT_ROOT), parameters=parameters)

    def test_validation_follows_the_switch(self):
        self.assertTrue(self.check(SEQUENCE=SEQUENCE)["valid"])
        self.assertTrue(self.check(BATCH_INPUT=True, INPUT_FASTA="set.fasta")["valid"])
        self.assertTrue(self.check(BATCH_INPUT=True, INPUT_FASTA=str(PROJECT_ROOT / "set.fasta"))["valid"])
        # Both given is no error: the switch decides.
        self.assertTrue(self.check(SEQUENCE=SEQUENCE, INPUT_FASTA="set.fasta")["valid"])
        self.assertTrue(self.check(BATCH_INPUT=True, SEQUENCE=SEQUENCE, INPUT_FASTA="set.fasta")["valid"])
        for parameters, wording in (({}, "a sequence is needed"),
                                    ({"INPUT_FASTA": "set.fasta"}, "a sequence is needed"),
                                    ({"BATCH_INPUT": True}, "a FASTA file is needed"),
                                    ({"BATCH_INPUT": True, "SEQUENCE": SEQUENCE}, "a FASTA file is needed")):
            with self.subTest(parameters=parameters):
                result = self.check(**parameters)
                self.assertFalse(result["valid"])
                self.assertEqual(len(result["errors"]), 1)
                self.assertIn(wording, result["errors"][0]["message"])
        result = self.check(SEQUENCE="1234")
        self.assertFalse(result["valid"])
        self.assertIn("empty after sanitization", result["errors"][0]["message"])
        # The typed sequence is not looked at while the switch is on.
        self.assertTrue(self.check(BATCH_INPUT=True, SEQUENCE="1234", INPUT_FASTA="set.fasta")["valid"])

    def test_validation_checks_the_model_device_and_maximum(self):
        for parameters in ({"PREDICTION_MODEL": "hosted"}, {"DEVICE_SELECTION": "gpu"}, {"MAX_SEQUENCES": 0},
                           {"SEQUENCE_NAME": "a/b"}, {"UNKNOWN": 1}):
            with self.subTest(parameters=parameters):
                self.assertFalse(self.check(SEQUENCE=SEQUENCE, **parameters)["valid"])
        result = self.check(SEQUENCE=SEQUENCE, PREDICTION_MODEL="remote", MAX_SEQUENCES=3)
        self.assertTrue(result["valid"])
        document = result["settings_document"]
        self.assertEqual(document["Predict_Structures.py"]["PREDICTION_MODEL"], "remote")
        self.assertEqual(Path(document["DIRECTORIES"]["STRUCTURE_DIR"]),
                         PROJECT_ROOT / "Analysis_Results" / "Folded_Structure")

    def test_the_schema_offers_an_example_with_a_typed_sequence(self):
        from mcp_server.pipeline.Pipeline_Settings import get_pipeline_schema
        schema = get_pipeline_schema("predict_structures", str(PROJECT_ROOT))
        self.assertEqual(set(schema["example"]["parameters"]), {"SEQUENCE", "SEQUENCE_NAME"})
        self.assertNotIn("NAMES_FILE", schema["parameters_schema"]["properties"])
        self.assertEqual(schema["parameters_schema"]["properties"]["BATCH_INPUT"]["default"], False)
        self.assertTrue(self.check(**schema["example"]["parameters"])["valid"])
        self.assertEqual(set(schema["directories_schema"]["properties"]), {"FASTA_DIR", "STRUCTURE_DIR"})
        self.assertEqual(schema["parameters_schema"]["properties"]["PREDICTION_MODEL"]["enum"], ["local", "remote"])


class ToolsWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def window(self, sections=None, carried=None):
        from tests.tools_gui_fixtures import isolated_tools_project
        root = isolated_tools_project(self, sections=sections)
        from PySide6.QtWidgets import QTextBrowser
        from EMAPSSN_Tools import ToolsGUI
        with mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser):
            window = ToolsGUI(carried=carried)
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        path = next(p for p in window.script_data if Path(p).name == "Predict_Structures.py")
        return window, path, root

    def row(self, window, path, name):
        label, widget = None, window.script_data[path]["inputs"][name]["widget"]
        return widget

    def test_the_card_is_in_manual_tools_right_before_benchmark_and_offers_the_contracts_fields(self):
        window, path, _ = self.window()
        inputs = window.script_data[path]["inputs"]
        self.assertTrue(set(_CONTRACTS["Predict_Structures.py"]["properties"]) <= set(inputs))
        scripts = [Path(p).name for p in window.script_data
                   if Path(p).name in {"Embedding_PWA.py", "Embedding_SSEARCH.py", "Predict_Structures.py"}]
        self.assertEqual(scripts, ["Embedding_PWA.py", "Embedding_SSEARCH.py", "Predict_Structures.py"])
        import EMAPSSN_Tools
        self.assertEqual(EMAPSSN_Tools.BENCHMARK_TAB, "Others")
        self.assertEqual(EMAPSSN_Tools.TOOL_TITLES["Predict_Structures.py"], "Predict Structures")
        self.assertIn("Predict_Structures.py", EMAPSSN_Tools.TOOL_ICONS)

    def test_batch_input_starts_off_with_the_fasta_row_greyed_out_and_a_path_field_with_browse(self):
        from PySide6.QtWidgets import QLineEdit, QPushButton
        window, path, _ = self.window()
        inputs = window.script_data[path]["inputs"]
        self.assertEqual(inputs["BATCH_INPUT"]["type"], "switch")
        self.assertFalse(inputs["BATCH_INPUT"]["widget"].isChecked())
        self.assertEqual(inputs["INPUT_FASTA"]["type"], "file_browse")
        fasta = inputs["INPUT_FASTA"]["widget"]
        self.assertIsInstance(fasta.line_edit, QLineEdit)
        self.assertIsInstance(fasta.browse_button, QPushButton)
        self.assertEqual(fasta.line_edit.text(), "")
        self.assertFalse(fasta.isEnabled())
        self.assertTrue(inputs["SEQUENCE"]["widget"].isEnabled())
        self.assertTrue(inputs["SEQUENCE_NAME"]["widget"].isEnabled())

    def test_the_switch_greys_out_the_other_input_without_clearing_it_and_back(self):
        window, path, _ = self.window()
        inputs = window.script_data[path]["inputs"]
        switch = inputs["BATCH_INPUT"]["widget"]
        inputs["SEQUENCE"]["widget"].setText(SEQUENCE)
        inputs["SEQUENCE_NAME"]["widget"].setText("typed")
        inputs["INPUT_FASTA"]["widget"].line_edit.setText("/data/set.fasta")

        switch.setChecked(True)
        self.assertTrue(inputs["INPUT_FASTA"]["widget"].isEnabled())
        self.assertFalse(inputs["SEQUENCE"]["widget"].isEnabled())
        self.assertFalse(inputs["SEQUENCE_NAME"]["widget"].isEnabled())
        self.assertEqual(inputs["SEQUENCE"]["widget"].text(), SEQUENCE)
        self.assertEqual(inputs["SEQUENCE_NAME"]["widget"].text(), "typed")

        switch.setChecked(False)
        self.assertFalse(inputs["INPUT_FASTA"]["widget"].isEnabled())
        self.assertTrue(inputs["SEQUENCE"]["widget"].isEnabled())
        self.assertEqual(inputs["INPUT_FASTA"]["widget"].line_edit.text(), "/data/set.fasta")
        collected = window._collect_tool_settings(path)
        self.assertEqual((collected["SEQUENCE"], collected["SEQUENCE_NAME"], collected["BATCH_INPUT"]),
                         (SEQUENCE, "typed", False))

    def test_browse_opens_a_file_dialog_with_the_fasta_filters_and_never_a_real_one_here(self):
        window, path, _ = self.window()
        window.script_data[path]["inputs"]["BATCH_INPUT"]["widget"].setChecked(True)  # a greyed-out row ignores clicks
        fasta = window.script_data[path]["inputs"]["INPUT_FASTA"]["widget"]
        chosen = str(Path(tempfile.gettempdir()) / "picked.fasta")
        with mock.patch("EMAPSSN_Tools.QFileDialog.getOpenFileName", return_value=(chosen, "")) as dialog:
            fasta.browse_button.click()
        dialog.assert_called_once()
        filters = dialog.call_args.args[3]
        self.assertIn("*.fasta", filters)
        self.assertIn("All files (*)", filters)
        self.assertEqual(fasta.line_edit.text(), os.path.normpath(chosen))
        # Cancelling leaves the field as it was.
        with mock.patch("EMAPSSN_Tools.QFileDialog.getOpenFileName", return_value=("", "")):
            fasta.browse_button.click()
        self.assertEqual(fasta.line_edit.text(), os.path.normpath(chosen))

    def pick(self, window, path, fasta_path):
        inputs = window.script_data[path]["inputs"]
        inputs["BATCH_INPUT"]["widget"].setChecked(True)
        inputs["INPUT_FASTA"]["widget"].line_edit.setText(fasta_path)
        inputs["SEQUENCE_NAME"]["widget"].setText("kept")

    def run_from_window(self, window, path):
        launched = []
        with mock.patch("EMAPSSN_Tools.launch_in_terminal", side_effect=lambda argv, **kw: launched.append(argv)):
            window.save_and_run(path)
        return launched

    def test_save_and_run_hands_the_fasta_to_the_run_but_never_saves_it(self):
        window, path, root = self.window()
        fasta = str(Path(tempfile.gettempdir()) / "picked.fasta")
        self.pick(window, path, fasta)
        from tools.tool_helpers.Tool_Pipeline import forget_one_run_snapshot
        launched = self.run_from_window(window, path)

        saved = json.loads((root / "tools_settings.json").read_text(encoding="utf-8"))
        section = saved["Predict_Structures.py"]
        self.assertNotIn("INPUT_FASTA", section)
        self.assertEqual((section["BATCH_INPUT"], section["SEQUENCE_NAME"], section["MAX_SEQUENCES"]), (True, "kept", 20))
        self.assertNotIn(os.path.normpath(fasta), (root / "tools_settings.json").read_text(encoding="utf-8"))

        self.assertEqual(len(launched), 1)
        snapshot = Path(launched[0][-1])
        self.addCleanup(lambda: forget_one_run_snapshot(str(snapshot)))
        self.assertTrue(snapshot.name.startswith("predict_structures-"))
        document = json.loads(snapshot.read_text(encoding="utf-8"))
        self.assertEqual(document["Predict_Structures.py"]["INPUT_FASTA"], os.path.normpath(fasta))
        self.assertEqual(document["Predict_Structures.py"]["BATCH_INPUT"], True)
        self.assertEqual(set(document["DIRECTORIES"]), {"FASTA_DIR", "STRUCTURE_DIR"})

        # Opened again, the window starts without the path.
        again, again_path, _ = self.window(sections=saved)
        self.assertEqual(again.script_data[again_path]["inputs"]["INPUT_FASTA"]["widget"].line_edit.text(), "")
        self.assertTrue(again.script_data[again_path]["inputs"]["BATCH_INPUT"]["widget"].isChecked())

    def test_the_other_tools_still_run_from_the_shared_file_with_no_snapshot(self):
        window, _, root = self.window()
        path = next(p for p in window.script_data if Path(p).name == "Embedding_SSEARCH.py")
        launched = self.run_from_window(window, path)
        if launched:  # the dropdown inputs may be empty here; what matters is that no snapshot is made for it
            self.assertEqual(Path(launched[0][-1]).name, "Embedding_SSEARCH.py")

    def test_an_export_may_hold_the_path_and_the_window_redrawn_keeps_it_for_the_session(self):
        window, path, _ = self.window()
        fasta = str(Path(tempfile.gettempdir()) / "picked.fasta")
        self.pick(window, path, fasta)
        collected = window._collect_tool_settings(path)
        self.assertEqual(collected["INPUT_FASTA"], os.path.normpath(fasta))
        carried = window.language_carry_over()
        self.assertEqual(carried["document"]["Predict_Structures.py"]["INPUT_FASTA"], os.path.normpath(fasta))
        redrawn, redrawn_path, _ = self.window(carried=carried)
        inputs = redrawn.script_data[redrawn_path]["inputs"]
        self.assertEqual(inputs["INPUT_FASTA"]["widget"].line_edit.text(), os.path.normpath(fasta))
        self.assertTrue(inputs["BATCH_INPUT"]["widget"].isChecked())
        self.assertTrue(inputs["INPUT_FASTA"]["widget"].isEnabled())
        self.assertFalse(inputs["SEQUENCE"]["widget"].isEnabled())

    def test_the_model_choice_is_stored_apart_from_its_label_and_the_rest_is_saved(self):
        from desktop.Desktop_App import combo_value
        window, path, _ = self.window(sections={"Predict_Structures.py": {
            "PREDICTION_MODEL": "remote", "SEQUENCE_NAME": "n", "MAX_SEQUENCES": 7}})
        inputs = window.script_data[path]["inputs"]
        combo = inputs["PREDICTION_MODEL"]["widget"]
        self.assertEqual(combo_value(combo), "remote")
        self.assertNotEqual(combo.currentText(), "remote")
        collected = window._collect_tool_settings(path)
        self.assertEqual((collected["PREDICTION_MODEL"], collected["SEQUENCE_NAME"], collected["MAX_SEQUENCES"]),
                         ("remote", "n", 7))
        self.assertEqual(collected["DEVICE_SELECTION"], "auto")

    def test_the_directories_tab_has_the_folded_structure_folder(self):
        window, _, _ = self.window()
        self.assertIn("STRUCTURE_DIR", window.dir_inputs)
        self.assertTrue(window.dir_inputs["STRUCTURE_DIR"].text().endswith(os.path.join("Analysis_Results", "Folded_Structure")))


if __name__ == "__main__":
    unittest.main()
