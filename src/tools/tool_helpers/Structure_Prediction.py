# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
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

"""Predict protein structures with ESM3 for sequences given directly or read from a FASTA.

This is the engine of the Predict Structures tool, with no window and no Viewer. The model
runs in the Viewer's fold worker (``resources.esmfold.esmfold_worker.run_predictions``), so
a structure is written exactly as the Viewer's ``esmfold`` command writes it, into the tool's
own folder.

``predict_structures`` takes ``(name, sequence)`` pairs and returns one StructureRecord for
each; a caller that has sequences already (rather than a FASTA) gives them directly. It
refuses a request that cannot be run before anything is created, by raising
StructureRequestError with one line that says why.
"""

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass

from resources import Biohub_API
from resources.esmfold import esmfold_worker
from utilities.Sequence_Utils import read_fasta, sanitize_header, sanitize_sequence
from web_ui.residue_layers import STANDARD_LETTERS, THREE_TO_ONE


LOCAL = "local"
REMOTE = "remote"
MODES = (LOCAL, REMOTE)

FOLDED = "folded"
EXISTING = "existing"
FAILED = "failed"

# More sequences than this to fold need the limit raised or a smaller FASTA, so that a whole large
# FASTA, or the credit of the remote service, is not spent by accident.
DEFAULT_MAX_SEQUENCES = 20

# A structure's file name is its sequence's name, the remote model's name when there is one, and ".pdb".
# A name longer than this keeps its beginning and ends in "_" and the first hex digits of the SHA-256 of
# the whole name, so two long names that start alike stay different files.
MAX_STEM_LENGTH = 100
STEM_HASH_LENGTH = 8
MIN_STEM_LENGTH = 24
# The longest path Windows opens without a long-path setting, with the room for the end of it.
WINDOWS_PATH_LIMIT = 259

# The MCP server's job for this tool names a file in the job's own folder (never the structures folder) that the
# records are written to as JSON; a run by hand or from the Tools window has no such variable and writes none.
RESULT_FILE_VARIABLE = "SSN_TOOL_RESULT_FILE"

GENERATED_NAME_PREFIX = "sequence_"
GENERATED_NAME_HEXDIGITS = 10
# What a note says about a structure whose residues could not be read to compare with the sequence.
UNCHECKED_NOTE = "its sequence was not checked: the file holds no readable residues"


class StructureRequestError(ValueError):
    """A request that cannot be run: str(error) is the one line to show."""


class StructureRunError(RuntimeError):
    """The model could not start or the run ended early: str(error) is the one line to show."""


@dataclass(frozen=True)
class StructureRecord:
    """What became of one sequence: ``status`` is FOLDED, EXISTING or FAILED."""

    name: str
    path: str
    length: int
    status: str
    mean_plddt: float | None = None
    note: str = ""


@dataclass(frozen=True)
class _Planned:
    name: str
    sequence: str
    stem: str
    path: str


# --- Names and sequences ---

def generated_name(sequence):
    """The name of a sequence given without one: it comes from the sequence itself, so the same
    sequence finds its structure again and another one never takes the name."""
    digest = hashlib.sha256(sequence.encode("ascii", errors="replace")).hexdigest()
    return GENERATED_NAME_PREFIX + digest[:GENERATED_NAME_HEXDIGITS]


def clean_sequence(value, label="sequence"):
    """The sequence as the other tools clean a typed one (utilities.Sequence_Utils.sanitize_sequence):
    uppercase, with the ends trimmed to residues and any other character inside masked as X."""
    cleaned, _, _ = sanitize_sequence(str(value or ""))
    if not cleaned:
        raise StructureRequestError(f"The {label} holds no amino acid residues after sanitization.")
    return cleaned


def sequence_request(sequence, name=""):
    """The one ``(name, sequence)`` of a sequence given directly, with an optional name."""
    cleaned = clean_sequence(sequence)
    name = str(name or "").strip()
    if not name:
        return [(generated_name(cleaned), cleaned)]
    safe_name, _ = sanitize_header(name)
    if not safe_name.strip("_"):
        raise StructureRequestError(f"The sequence name {name!r} holds no usable characters.")
    return [(safe_name, cleaned)]


def _listed(names, noun=None):
    shown = ", ".join(repr(name) for name in names[:5])
    return shown + (f" and {len(names) - 5} more" if len(names) > 5 else "")


def fasta_requests(fasta_path):
    """The ``(name, sequence)`` pairs of the FASTA at ``fasta_path``.

    A sequence's name is its header made safe as the other tools make a header (spaces become
    underscores, characters that tools cannot take become underscores).
    """
    headers, sequences = read_fasta(fasta_path)
    if not headers:
        raise StructureRequestError(f"The FASTA file holds no sequences: {fasta_path}")
    entries = [(sanitize_header(header)[0], sequence) for header, sequence in zip(headers, sequences)]
    requests, empty = [], []
    for name, sequence in entries:
        cleaned, _, _ = sanitize_sequence(sequence)
        if cleaned:
            requests.append((name, cleaned))
        else:
            empty.append(name)
    if empty:
        raise StructureRequestError(
            f"{len(empty)} sequence(s) hold no amino acid residues after sanitization: {_listed(empty, 'sequence')}.")
    return requests


# --- File names ---

def _stem(name, limit):
    stem = esmfold_worker.sanitize_filename(name)
    if len(stem) <= limit:
        return stem
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:STEM_HASH_LENGTH]
    return stem[:limit - STEM_HASH_LENGTH - 1] + "_" + digest


def _stem_limit(structures_dir, suffix, path_limit):
    limit = MAX_STEM_LENGTH
    if path_limit:
        room = path_limit - len(structures_dir) - 1 - len(".pdb") - (len(suffix) + 1 if suffix else 0)
        limit = min(limit, room)
        if limit < MIN_STEM_LENGTH:
            raise StructureRequestError(
                f"The structure folder's path is too long to save structure files in: {structures_dir}")
    return limit


def _plan(requests, structures_dir, suffix, path_limit):
    limit = _stem_limit(structures_dir, suffix, path_limit)
    planned, claimed = [], {}
    for name, sequence in requests:
        stem = _stem(name, limit)
        filename = esmfold_worker.prediction_filename(stem, suffix)
        if filename in claimed:
            raise StructureRequestError(
                f"The sequence names {claimed[filename]!r} and {name!r} give the same structure file name, "
                f"{filename}; give them different names.")
        claimed[filename] = name
        planned.append(_Planned(name, sequence, stem, os.path.join(structures_dir, filename)))
    return planned


# --- Structure files ---

def read_structure(path):
    """``(the residues of the first chain as one-letter codes, the mean pLDDT)`` of the PDB at ``path``.

    ESM writes each residue's pLDDT, on a scale of 0 to 100, as the B-factor of its atoms; the mean is
    over the alpha carbons. A residue the PDB names other than the twenty (UNK, SEC) reads as X. Either
    part is None when the file does not give it reliably: the sequence of a file with no residues, or
    with more than one chain, and the mean when an alpha carbon's B-factor is not a number.
    """
    residues, chains, factors = [], set(), []
    last = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("ATOM  "):
                    continue
                chains.add(line[21])
                residue = (line[21], line[22:27])
                if residue != last:
                    residues.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
                    last = residue
                if line[12:16].strip() == "CA":
                    try:
                        factors.append(float(line[60:66]))
                    except ValueError:
                        factors = None
                        break
    except OSError:
        return None, None
    sequence = "".join(residues) if residues and len(chains) == 1 else None
    plddt = round(sum(factors) / len(factors), 1) if factors else None
    return sequence, plddt


def _same_protein(requested, found):
    """Whether the residues of a file are the requested sequence: as long, and every standard residue of
    one is the other's (a letter that is no standard amino acid, such as the X of a masked residue, is
    not compared)."""
    if len(requested) != len(found):
        return False
    return all(a == b or a not in STANDARD_LETTERS or b not in STANDARD_LETTERS
               for a, b in zip(requested, found))


def _saved(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


# --- Device and credentials ---

def _check_remote():
    state, error = Biohub_API.credentials_state(Biohub_API.API_SETTINGS_FILE)
    if state == "invalid":
        raise StructureRequestError(f"The Biohub API settings cannot be used: {error}")
    if state == "missing":
        raise StructureRequestError(
            "Remote prediction needs a saved Biohub API token. Save one by running "
            "esmfold large in the Viewer once, or set ESM_API_KEY; the model setting is remote, "
            "so nothing is sent and the local model is not used instead.")
    return Biohub_API.remote_model_name()


def _check_device(selection):
    """The device the local model runs on, as a name torch takes; ValueError says if the selection is not one."""
    from utilities import Hardware_Acceleration as hardware
    try:
        candidate = hardware.resolve_device_selection(selection, hardware.get_available_devices())
    except ValueError as error:
        raise StructureRequestError(str(error)) from error
    return candidate.spec if candidate is not None else str(hardware.get_optimal_device())


def _refuse_to_ask():
    raise Biohub_API.BiohubSettingsError(
        "Biohub API credentials are missing; nothing can be asked here. Save a token or set ESM_API_KEY.")


# --- The run ---

def predict_structures(requests, structures_dir, *, mode=LOCAL, device="auto",
                       max_sequences=DEFAULT_MAX_SEQUENCES, log=print, path_limit=None):
    """Predict the structure of each ``(name, sequence)`` of ``requests``; return a StructureRecord for each, in order.

    ``mode`` is LOCAL (the ESM3 open model, on the first available accelerator, else the CPU; ``device`` names
    one instead for tests: "cpu", "cuda:0", ... No setting of a tool offers it) or REMOTE (the
    Biohub API, with the model of the saved API settings). The remote service is used only when REMOTE is
    asked for, and the local model never takes its place. A structure file that exists is kept, never
    written again: it is reported EXISTING when it holds this sequence, and FAILED when it holds another.

    Raises StructureRequestError before anything is created when the request cannot be run (no sequences,
    names that give one file, more to fold than ``max_sequences``, an unavailable device, a remote run
    with no saved token), and StructureRunError when the model cannot start. ``log`` takes the lines the
    run prints.
    """
    if mode not in MODES:
        raise StructureRequestError(f"The model must be one of {', '.join(MODES)}, not {mode!r}.")
    requests = list(requests)
    if not requests:
        raise StructureRequestError("There are no sequences to predict.")
    try:
        max_sequences = int(max_sequences)
    except (TypeError, ValueError):
        raise StructureRequestError("The maximum number of sequences must be a whole number.") from None
    if max_sequences < 1:
        raise StructureRequestError("The maximum number of sequences must be at least 1.")
    suffix = None
    target_device = None
    if mode == REMOTE:
        suffix = _check_remote()
    else:
        target_device = _check_device(device)
    structures_dir = os.path.abspath(structures_dir)
    if path_limit is None and sys.platform == "win32":
        path_limit = WINDOWS_PATH_LIMIT
    planned = _plan(requests, structures_dir, suffix, path_limit)

    records = {}
    to_fold = []
    for item in planned:
        if not _saved(item.path):
            to_fold.append(item)
            continue
        found, plddt = read_structure(item.path)
        if found is None:
            records[item.name] = StructureRecord(item.name, item.path, len(item.sequence), EXISTING, plddt, UNCHECKED_NOTE)
        elif _same_protein(item.sequence, found):
            records[item.name] = StructureRecord(item.name, item.path, len(item.sequence), EXISTING, plddt)
        else:
            records[item.name] = StructureRecord(
                item.name, item.path, len(item.sequence), FAILED, None,
                f"a structure named {os.path.basename(item.path)} already exists for a different sequence; "
                "choose another name or output folder")
    if len(to_fold) > max_sequences:
        raise StructureRequestError(
            f"{len(to_fold)} sequences would be folded, more than the maximum of {max_sequences}. "
            "Raise the maximum number of sequences, or use a smaller FASTA.")

    if to_fold:
        _fold(to_fold, structures_dir, mode, target_device, suffix, records, log)
    return [records[item.name] for item in planned]


def _fold(to_fold, structures_dir, mode, target_device, suffix, records, log):
    by_stem = {item.stem: item for item in to_fold}
    reasons = {}

    def folded(stem, _filename):
        item = by_stem[stem]
        _, plddt = read_structure(item.path)
        records[item.name] = StructureRecord(item.name, item.path, len(item.sequence), FOLDED, plddt)

    def event(kind, **fields):
        if kind == "structure_failed":
            reasons[fields.get("name")] = fields.get("reason") or "the prediction failed"

    if mode == REMOTE:
        log(f"Sending {len(to_fold)} sequence(s) to the Biohub API for prediction (model {suffix}).")
    else:
        log(f"Predicting {len(to_fold)} sequence(s) with the local ESM3 open model on {target_device}.")
    try:
        with esmfold_worker.never_ask():
            esmfold_worker.run_predictions(
                [[item.stem, item.sequence] for item in to_fold],
                structures_dir,
                mode="large" if mode == REMOTE else "local",
                target_device=target_device,
                token_prompt=_refuse_to_ask,
                notifier=folded,
                skip_existing=False,
                job_event=event,
            )
    except Biohub_API.BiohubAuthenticationError as error:
        reasons.setdefault(None, f"Biohub rejected the saved API token (HTTP {error.error_code})")
    except Exception as error:
        if not any(item.name in records for item in to_fold):
            raise StructureRunError(str(error)) from error
        reasons.setdefault(None, str(error))
    for item in to_fold:
        if item.name not in records:
            reason = reasons.get(item.stem) or reasons.get(None) or "the prediction produced no structure"
            records[item.name] = StructureRecord(item.name, item.path, len(item.sequence), FAILED, None, reason)


def write_result_file(path, records, mode):
    """Write ``records`` as JSON to ``path``, whole or not at all: {"model": mode, "structures": [...]}."""
    document = {"model": mode, "structures": [
        {"name": record.name, "path": record.path, "length": record.length, "status": record.status,
         "mean_plddt": record.mean_plddt, "note": record.note} for record in records]}
    temporary = f"{path}.{os.getpid()}.partial"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    os.replace(temporary, path)


def failed(records):
    """The records of the sequences that did not end with a structure."""
    return [record for record in records if record.status == FAILED]


def format_table(records):
    """The lines of a table of ``records``: name, file, length, status and mean pLDDT, then each note."""
    rows = [("Name", "File", "Length", "Status", "Mean pLDDT")]
    for record in records:
        rows.append((record.name, os.path.basename(record.path), str(record.length), record.status,
                     "-" if record.mean_plddt is None else f"{record.mean_plddt:.1f}"))
    widths = [max(len(row[column]) for row in rows) for column in range(5)]
    lines = []
    for number, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(width) if column in (0, 1, 3) else cell.rjust(width)
                               for column, (cell, width) in enumerate(zip(row, widths))).rstrip())
        if number == 0:
            lines.append("  ".join("-" * width for width in widths))
    for record in records:
        if record.note:
            lines.append(f"Note: {record.name}: {record.note}.")
    return lines
