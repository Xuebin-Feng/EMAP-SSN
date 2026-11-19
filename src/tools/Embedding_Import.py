# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
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

"""
File: Embedding_Import.py
===================================
Description:
Converts per-residue embeddings computed elsewhere (UniProt's ProtT5 files, the
ProtTrans scripts, a model of your own) into an EMAP-SSN embedding file, so the
alignment, MSA and network tools can use them. No model is run.

The FASTA decides what goes in: it is sanitized as Sanitize Sequences does, and
every sanitized record must find one residues-by-features array in the source.

Input:
- INPUT_FASTA: the sequence set, in FASTA_DIR (or an absolute path). A relative
  SOURCE_PATH or ID_MAP_FILE also resolves inside FASTA_DIR.
- SOURCE_PATH: an HDF5 file with one dataset per protein at its top level, named
  by ID (a dataset's `original_id` attribute also counts as its ID), a `.npz`
  file, or a folder of `.npy` files named by ID.
- ID_MAP_FILE (optional): a file of two columns, a FASTA header (as it stands in
  the FASTA, its first word, or the sanitized header) and the source ID it
  should take. A `.csv` file is comma-separated (a value with a comma goes in
  double quotes); any other file is tab-separated. Blank lines and lines whose
  first column starts with `#` are skipped; a column heading line is not
  recognised, so put `#` before it. A column after the second is ignored, and
  a line with fewer than two columns stops the run with an error naming the
  file and line. A header given twice with different IDs is an error too. A
  byte-order mark at the start is dropped. The mapping is tried before the
  automatic rules; a record it does not list goes through them.

Matching, per record, the first rule that finds an ID in the source: the mapping
file; the original header; its first word; the sanitized header; a UniProt
accession (`sp|P12345|NAME` gives P12345). Every original header of records that
sanitization merged is tried. An ID that more than one record could take is not
used, and those records are reported as ambiguous.

Checks on every array: two-dimensional (a per-protein vector is refused), one row
per residue of the sanitized sequence (rows for start and end tokens are named
as the likely cause), one feature width for the whole file, finite values at the
chosen precision.

Output:
- `<FASTA stem>_[<MODEL_LABEL>]_embeddings.h5` in EMBED_DIR. It is marked complete
  only when every record has its array; otherwise it stays incomplete, readers
  refuse it, and running the tool again (with a mapping file, say) resumes it.
- `<FASTA stem>_[<MODEL_LABEL>]_import_report.tsv` beside it, listing each record
  that was not imported and why, when there is one.
"""

import ast
import csv
import json
import os
import re
try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap

import h5py
import numpy as np

from utilities.Sequence_Utils import (
    print_sanitization_result,
    read_fasta,
    sanitize_fasta_records,
    sanitize_header,
    sanitize_sequence,
)
from utilities.HDF5_Storage import (
    EmbeddingWriter,
    dtype_for_saving_mode,
    validate_model_label,
)
from tools.tool_helpers.Model_Plugins import discover_model_execution_modes


# Script configuration
INPUT_FASTA = None
SOURCE_PATH = None
ID_MAP_FILE = None
MODEL_LABEL = None
# Keep equal to Generate_Embeddings.SAVING_MODE (a test checks it): the Tools
# window reads each tool's default from its own literal.
SAVING_MODE = "float16"

from tools.tool_helpers.Tool_Pipeline import (
    forget_one_run_snapshot,
    inherited_settings_path,
    load_tool_settings,
    project_directory_defaults,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DIRECTORIES = project_directory_defaults(PROJECT_ROOT)
FASTA_DIR = _DEFAULT_DIRECTORIES["FASTA_DIR"]
EMBED_DIR = _DEFAULT_DIRECTORIES["EMBED_DIR"]

SETTINGS_FILE = inherited_settings_path(__file__) or os.path.join(PROJECT_ROOT, "tools_settings.json")

if __name__ != "__main__" and os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as settings_handle:
            all_settings = json.load(settings_handle)
        for key, value in all_settings.get("DIRECTORIES", {}).items():
            if key in globals() and value is not None and str(value).strip():
                if not os.path.isabs(str(value)):
                    value = os.path.normpath(os.path.join(PROJECT_ROOT, str(value)))
                globals()[key] = value
        for key, value in all_settings.get(os.path.basename(__file__), {}).items():
            if key in globals() and value is not None and str(value).strip():
                original = globals()[key]
                if isinstance(original, list) and isinstance(value, str):
                    try:
                        value = ast.literal_eval(value)
                    except (SyntaxError, ValueError):
                        pass
                globals()[key] = value
    except Exception as exc:
        print(f"Failed to load user settings: {exc}")

PLUGIN_DIR = os.path.join(PROJECT_ROOT, "src", "resources", "pLM_models")
HDF5_EXTENSIONS = (".h5", ".hdf5", ".hdf", ".he5")
UNIPROT_HEADER = re.compile(r"(?:sp|tr)\|([^|\s]+)\|")
# The automatic matching rules, in the order they are tried.
RULES = ("header", "first_word", "sanitized_header", "uniprot_accession")


class EmbeddingImportError(ValueError):
    """A problem that stops the import before the output file is touched."""


# ---------------------------------------------------------------- the source
class EmbeddingSource:
    """One protein's array per ID: an HDF5 file, a .npz file or a folder of .npy files."""

    def __init__(self, path):
        self.path = os.path.normpath(os.fspath(path))
        self._h5 = None
        self._npz = None
        self._files = {}
        self.keys = {}  # ID -> the source's own key for its array
        self.dtypes = set()
        if os.path.isdir(self.path):
            self.reader = "npy_folder"
            for name in sorted(os.listdir(self.path)):
                stem, extension = os.path.splitext(name)
                if extension.lower() == ".npy":
                    self._files[stem] = os.path.join(self.path, name)
                    self.keys[stem] = stem
            if not self.keys:
                raise EmbeddingImportError(f"The folder {self.path} holds no .npy files.")
            return
        if not os.path.isfile(self.path):
            raise EmbeddingImportError(f"Embedding source not found: {self.path}")
        extension = os.path.splitext(self.path)[1].lower()
        if extension == ".npy":
            raise EmbeddingImportError(
                "A .npy file holds one array. Choose the folder that holds one .npy "
                "file per protein, each named by its ID."
            )
        if extension not in HDF5_EXTENSIONS + (".npz",):
            raise EmbeddingImportError(
                f"{os.path.basename(self.path)} is not a source this tool reads: "
                "give an HDF5 file (.h5), a .npz file or a folder of .npy files."
            )
        try:
            if extension == ".npz":
                self.reader = "npz"
                self._npz = np.load(self.path, allow_pickle=False)
                self.keys = {name: name for name in self._npz.files}
            else:
                self.reader = "hdf5"
                self._open_hdf5()
            if not self.keys:
                raise EmbeddingImportError(f"{os.path.basename(self.path)} holds no arrays.")
        except BaseException:
            self.close()
            raise

    def _open_hdf5(self):
        self._h5 = h5py.File(self.path, "r")
        if "model_name" in self._h5.attrs and "embeddings" in self._h5:
            raise EmbeddingImportError(
                f"{os.path.basename(self.path)} is already an EMAP-SSN embedding file: "
                "use it directly, or Embedding Extraction for a subset of it."
            )
        aliases = {}
        for name, item in self._h5.items():
            if not isinstance(item, h5py.Dataset):
                continue
            self.keys[name] = name
            original = item.attrs.get("original_id")
            if isinstance(original, bytes):
                original = original.decode("utf-8", "replace")
            if isinstance(original, str) and original and original != name:
                aliases.setdefault(original, []).append(name)
        # bio_embeddings names datasets by a checksum and keeps the ID in original_id.
        for alias, names in aliases.items():
            if alias not in self.keys and len(names) == 1:
                self.keys[alias] = names[0]
        if not self.keys:
            raise EmbeddingImportError(
                f"{os.path.basename(self.path)} holds no datasets at its top level; "
                "expected one dataset per protein, named by its ID."
            )

    def load(self, key):
        if self._h5 is not None:
            array = self._h5[key][()]
        elif self._npz is not None:
            array = self._npz[key]
        else:
            array = np.load(self._files[key], allow_pickle=False)
        self.dtypes.add(str(np.asarray(array).dtype))
        return array

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        if self._npz is not None:
            self._npz.close()
            self._npz = None


# ---------------------------------------------------------------- matching
def read_id_map(path):
    """Read a two-column ID mapping (FASTA header, source ID).

    A .csv file is comma-separated, any other file tab-separated; both follow the csv
    module's quoting. A leading byte-order mark (Excel writes one) is dropped. A line whose
    first column is empty or starts with '#' is skipped; a column after the second is ignored.
    """
    delimiter, separator = (",", "comma") if str(path).lower().endswith(".csv") else ("\t", "tab")
    mapping = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            number = reader.line_num
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 2 or not row[1].strip():
                raise EmbeddingImportError(
                    f"Line {number} of {os.path.basename(path)} needs two {separator}-separated "
                    "columns: a FASTA header and a source ID."
                )
            key, source_id = row[0].strip(), row[1].strip()
            if mapping.get(key, source_id) != source_id:
                raise EmbeddingImportError(
                    f"Line {number} of {os.path.basename(path)} maps '{key}' to "
                    f"'{source_id}', but an earlier line maps it to '{mapping[key]}'."
                )
            mapping[key] = source_id
    return mapping


def _first_word(header):
    words = header.split()
    return words[0] if words else ""


def record_candidates(header, original_headers):
    """The (rule, ID) pairs to try for one sanitized record, in rule order."""
    candidates = []
    for rule in RULES:
        for original in original_headers:
            if rule == "header":
                value = original
            elif rule == "first_word":
                value = _first_word(original)
            elif rule == "sanitized_header":
                value = sanitize_header(original)[0]
            else:
                match = UNIPROT_HEADER.match(original)
                value = match.group(1) if match else ""
            if value and (rule, value) not in candidates:
                candidates.append((rule, value))
        if rule == "sanitized_header" and ("sanitized_header", header) not in candidates:
            candidates.append(("sanitized_header", header))
    return candidates


def match_records(headers, sequences, raw_headers, raw_sequences, source_keys, id_map=None):
    """Decide each sanitized record's source ID.

    Returns one dict per record: status "matched" (with id and rule), "missing",
    "ambiguous" or "unmapped" (with reason).
    """
    originals_by_sequence = {}
    for raw_header, raw_sequence in zip(raw_headers, raw_sequences):
        cleaned = sanitize_sequence(raw_sequence)[0]
        if cleaned:
            originals_by_sequence.setdefault(cleaned, []).append(raw_header)

    candidates = [
        record_candidates(header, originals_by_sequence.get(sequence, []))
        for header, sequence in zip(headers, sequences)
    ]
    takers = {}
    for index, record_ids in enumerate(candidates):
        for _rule, value in record_ids:
            takers.setdefault(value, set()).add(index)

    results = []
    for index, header in enumerate(headers):
        originals = originals_by_sequence.get(sequences[index], [])
        if id_map:
            keys = [header] + [
                key for original in originals
                for key in (original, _first_word(original), sanitize_header(original)[0])
            ]
            mapped = next((id_map[key] for key in keys if key in id_map), None)
            if mapped is not None:
                if mapped in source_keys:
                    results.append({"status": "matched", "id": mapped, "rule": "id_map"})
                else:
                    results.append({
                        "status": "unmapped", "id": mapped,
                        "reason": f"the mapping file names source ID '{mapped}', which the source lacks",
                    })
                continue
        chosen, shared = None, []
        for rule, value in candidates[index]:
            if value not in source_keys:
                continue
            if len(takers[value]) > 1:
                shared.append(value)
                continue
            chosen = {"status": "matched", "id": value, "rule": rule}
            break
        if chosen:
            results.append(chosen)
        elif shared:
            results.append({
                "status": "ambiguous", "id": shared[0],
                "reason": f"source ID '{shared[0]}' fits more than one FASTA record; map it in an ID mapping file",
            })
        else:
            results.append({"status": "missing", "id": "", "reason": "no ID of this record is in the source"})
    return results


# ---------------------------------------------------------------- the import
def shipped_model_ids():
    return set(discover_model_execution_modes(PLUGIN_DIR))


def output_paths(fasta_path, model_label, embed_dir):
    stem = os.path.splitext(os.path.basename(fasta_path))[0]
    base = os.path.join(embed_dir, f"{stem}_[{model_label}]")
    return base + "_embeddings.h5", base + "_import_report.tsv"


def check_label(model_label):
    validate_model_label(model_label)
    if model_label in shipped_model_ids():
        raise EmbeddingImportError(
            f"'{model_label}' names a model EMAP-SSN computes itself, so an imported file "
            "under it would pass for one computed here, and Embedding Injection would add "
            "locally computed embeddings to it. Give imported embeddings a label of their "
            f"own, such as 'imported_{model_label}'."
        )
    return model_label


def import_embeddings(fasta_path, source_path, model_label, saving_mode, embed_dir, id_map_path=None):
    """Convert the source's arrays for the FASTA's sanitized records into an embedding file."""
    check_label(model_label)
    dtype_for_saving_mode(saving_mode)
    if not os.path.isfile(fasta_path):
        raise EmbeddingImportError(f"FASTA file not found: {fasta_path}")
    raw_headers, raw_sequences = read_fasta(fasta_path)
    headers, sequences, stats = sanitize_fasta_records(raw_headers, raw_sequences)
    print_sanitization_result(stats)
    if not headers:
        raise EmbeddingImportError(f"{os.path.basename(fasta_path)} holds no sequences.")
    id_map = read_id_map(id_map_path) if id_map_path else None
    output_hdf5, report_path = output_paths(fasta_path, model_label, embed_dir)

    source = EmbeddingSource(source_path)
    try:
        results = match_records(headers, sequences, raw_headers, raw_sequences, source.keys, id_map)
        # No ID of any record in the source at all: most likely the wrong source or headers,
        # so nothing is created. Ambiguous or mapped records still get the file and its report.
        if all(r["status"] == "missing" for r in results) and not os.path.exists(output_hdf5):
            examples = ", ".join(list(source.keys)[:3])
            raise EmbeddingImportError(
                f"No record of {os.path.basename(fasta_path)} found its ID in the source "
                f"(source IDs look like: {examples}; FASTA headers look like: "
                f"{', '.join(raw_headers[:3])}). Give an ID mapping file."
            )
        imported_now = 0
        rule_counts = {}
        with EmbeddingWriter(output_hdf5, (headers, sequences), model_label, saving_mode) as writer:
            pending = {header for header, _ in writer.pending}
            already = len(headers) - len(pending)
            for header, sequence, result in zip(headers, sequences, results):
                if header not in pending or result["status"] != "matched":
                    continue
                try:
                    writer.add(header, source.load(source.keys[result["id"]]))
                except (ValueError, TypeError, OSError, KeyError) as error:
                    result.update(status="invalid", reason=str(error).strip("'\""))
                    continue
                imported_now += 1
                rule_counts[result["rule"]] = rule_counts.get(result["rule"], 0) + 1
            if imported_now:
                # Where this file's arrays came from; a run that adds nothing leaves the record alone.
                writer.set_attributes(
                    import_source=os.path.basename(source.path),
                    import_reader=source.reader,
                    import_source_dtype=",".join(sorted(source.dtypes)) or "unknown",
                    import_matched_by=json.dumps(rule_counts, sort_keys=True),
                )
            remaining = [header for header, _ in writer.pending]
        complete = not remaining
    finally:
        source.close()

    remaining = set(remaining)
    problems = [
        (header, result) for header, result in zip(headers, results)
        if header in remaining
    ]
    if problems:
        with open(report_path, "w", encoding="utf-8", newline="") as handle:
            writer_tsv = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer_tsv.writerow(["header", "status", "source_id", "reason"])
            for header, result in problems:
                writer_tsv.writerow([header, result["status"], result.get("id", ""), result.get("reason", "")])
    elif os.path.exists(report_path):
        os.remove(report_path)  # an earlier run's list, no longer true

    used_keys = {source.keys[r["id"]] for r in results if r["status"] in ("matched", "invalid")}
    counts = {status: sum(r["status"] == status for _, r in problems) for status in ("missing", "ambiguous", "unmapped", "invalid")}
    return {
        "output_hdf5": output_hdf5,
        "report_path": report_path if problems else None,
        "complete": complete,
        "records": len(headers),
        "imported_now": imported_now,
        "already_present": already,
        "matched_by": rule_counts,
        "unused_source_ids": len(set(source.keys.values()) - used_keys),
        "reader": source.reader,
        **counts,
    }


def _resolve(path, directory):
    if not path or not str(path).strip():
        return ""
    path = os.path.expanduser(str(path).strip())
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(directory, path))


def main(argv=None):
    # The Tools window does not save the two paths picked with Browse; it hands this run a document of its own.
    forget_one_run_snapshot(load_tool_settings(globals(), __file__, PROJECT_ROOT, argv))
    print("--- Embedding Import ---")
    try:
        for name, value in (("INPUT_FASTA", INPUT_FASTA), ("SOURCE_PATH", SOURCE_PATH), ("MODEL_LABEL", MODEL_LABEL)):
            if not value or not str(value).strip():
                raise EmbeddingImportError(f"{name} is not set.")
        result = import_embeddings(
            _resolve(INPUT_FASTA, FASTA_DIR),
            _resolve(SOURCE_PATH, FASTA_DIR),
            str(MODEL_LABEL).strip(),
            SAVING_MODE,
            EMBED_DIR,
            id_map_path=_resolve(ID_MAP_FILE, FASTA_DIR) or None,
        )
    except (ValueError, OSError) as error:
        print(f"❌ Error: {error}")
        return 1

    print(f"\nSource: {result['reader']}; {result['records']} FASTA records.")
    print(f"  > Imported now: {result['imported_now']}"
          + (f" ({', '.join(f'{rule} {n}' for rule, n in sorted(result['matched_by'].items()))})" if result["matched_by"] else ""))
    if result["already_present"]:
        print(f"  > Already in the file: {result['already_present']}")
    for status, label in (("missing", "Missing from the source"), ("ambiguous", "Ambiguous IDs"),
                          ("unmapped", "Mapped to an absent ID"), ("invalid", "Refused arrays")):
        if result[status]:
            print(f"  > {label}: {result[status]}")
    print(f"  > Source IDs not used: {result['unused_source_ids']}")
    if result["complete"]:
        print(f"\n✅ Done! Embeddings saved to {result['output_hdf5']}")
        return 0
    print(f"\nThe records not imported are listed in {result['report_path']}.")
    print(f"❌ Error: {result['output_hdf5']} is incomplete, so no tool will read it yet. "
          "Fix the listed records (an ID mapping file helps), then run Embedding Import again to finish it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
