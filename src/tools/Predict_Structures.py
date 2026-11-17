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

"""
File: Predict_Structures.py
===================================
Description:
Predicts the 3D structure of protein sequences with ESM3, with no Viewer and no network file.

BATCH_INPUT chooses the input:
- Off (the default): SEQUENCE, one amino acid sequence typed or pasted in, with an optional
  SEQUENCE_NAME. With no name, the structure is named from the sequence itself ("sequence_" and the first
  10 hexadecimal digits of its SHA-256), so the same sequence finds its structure again. INPUT_FASTA is
  ignored.
- On: INPUT_FASTA, a FASTA file whose sequences are all folded. The Tools window picks it with a Browse
  button and does not save it; in a settings file it is an absolute path, or a name inside the FASTA
  directory. SEQUENCE and SEQUENCE_NAME are ignored.
The input that is not chosen is never read; the chosen one missing is refused before anything is created.

PREDICTION_MODEL chooses the model: "local" runs the ESM3 open model (biohub/esm3-sm-open-v1) on
DEVICE_SELECTION; "remote" sends the sequences to the Biohub API and needs the API token saved
beforehand. The remote service is used only when the setting says remote, and the local model never
takes its place.

Each structure is a PDB file in STRUCTURE_DIR, named like the Viewer's esmfold command names it. Its
B-factor column holds the per-residue pLDDT on a scale of 0 to 100. A structure file that exists is kept,
never overwritten. Nothing else is written beside the structures.
"""

import os

try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap

from tools.tool_helpers.Tool_Pipeline import (
    forget_one_run_snapshot,
    load_tool_settings,
    project_directory_defaults,
)
from tools.tool_helpers import Structure_Prediction as prediction

BATCH_INPUT = False
INPUT_FASTA = None
SEQUENCE = ""
SEQUENCE_NAME = ""
PREDICTION_MODEL = "local"
DEVICE_SELECTION = "auto"
MAX_SEQUENCES = 20  # prediction.DEFAULT_MAX_SEQUENCES; a literal, as the Tools window reads the defaults from the source

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DIRECTORIES = project_directory_defaults(PROJECT_ROOT)
FASTA_DIR = _DEFAULT_DIRECTORIES["FASTA_DIR"]
STRUCTURE_DIR = _DEFAULT_DIRECTORIES["STRUCTURE_DIR"]


def build_requests(batch_input, sequence, sequence_name, input_fasta, fasta_dir):
    """The ``(name, sequence)`` pairs the settings ask for; StructureRequestError says which input is missing.

    The switch decides: only the chosen input is looked at.
    """
    if batch_input:
        fasta = str(input_fasta or "").strip()
        if not fasta:
            raise prediction.StructureRequestError(
                "Batch Input is on, so a FASTA file is needed: set INPUT_FASTA, or turn Batch Input off to fold a typed sequence.")
        return prediction.fasta_requests(os.path.join(fasta_dir, fasta))
    typed = str(sequence or "").strip()
    if not typed:
        raise prediction.StructureRequestError(
            "Batch Input is off, so a sequence is needed: set SEQUENCE, or turn Batch Input on to fold a FASTA file.")
    return prediction.sequence_request(typed, sequence_name)


def main(argv=None):
    # The Tools window hands this tool a document for one run (it does not save the FASTA path); it is read once.
    forget_one_run_snapshot(load_tool_settings(globals(), __file__, PROJECT_ROOT, argv))
    print("--- Predict Structures ---")
    try:
        requests = build_requests(BATCH_INPUT, SEQUENCE, SEQUENCE_NAME, INPUT_FASTA, FASTA_DIR)
        records = prediction.predict_structures(
            requests,
            STRUCTURE_DIR,
            mode=PREDICTION_MODEL,
            device=DEVICE_SELECTION,
            max_sequences=MAX_SEQUENCES,
        )
    except (prediction.StructureRequestError, prediction.StructureRunError, FileNotFoundError) as error:
        print(f"❌ Error: {error}")
        return 1
    print()
    for line in prediction.format_table(records):
        print(line)
    print(f"\nStructures are in: {os.path.abspath(STRUCTURE_DIR)}")
    broken = prediction.failed(records)
    if broken:
        print(f"❌ {len(broken)} of {len(records)} sequence(s) failed.")
        return 1
    print(f"✅ Done: {len(records)} structure(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
