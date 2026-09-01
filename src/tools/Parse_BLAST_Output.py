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

"""Convert strict external BLAST tabular output into a viewer HDF5 network."""

import os

try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap

from utilities.BLAST_Tabular import BlastParseError, build_blast_network
from utilities.Tool_Directories import project_directory_defaults
from utilities.Tool_Settings import load_tool_settings


INPUT_BLAST_TABULAR = None
INPUT_FASTA = None
BLAST_LAYOUT = "standard_outfmt6"
QUERY_COLUMN = 1
SUBJECT_COLUMN = 2
EVALUE_COLUMN = 11

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DIRECTORIES = project_directory_defaults(PROJECT_ROOT)
FASTA_DIR = _DEFAULT_DIRECTORIES["FASTA_DIR"]
NETWORK_DIR = _DEFAULT_DIRECTORIES["NETWORK_DIR"]

FULL_INPUT_BLAST_TABULAR = None
FULL_INPUT_FASTA = None
OUTPUT_HDF5 = None


def _resolve_selected_path(selected, base_directory, label):
    if selected is None or not str(selected).strip():
        raise BlastParseError(f"No {label} has been selected.")
    selected = os.fspath(selected)
    return (
        os.path.normpath(selected)
        if os.path.isabs(selected)
        else os.path.normpath(os.path.join(base_directory, selected))
    )


def configure_runtime_paths():
    """Resolve GUI-selected BLAST and FASTA inputs immediately before parsing."""
    global FULL_INPUT_BLAST_TABULAR, FULL_INPUT_FASTA, OUTPUT_HDF5

    FULL_INPUT_BLAST_TABULAR = _resolve_selected_path(
        INPUT_BLAST_TABULAR, NETWORK_DIR, "BLAST tabular file"
    )
    FULL_INPUT_FASTA = _resolve_selected_path(
        INPUT_FASTA, FASTA_DIR, "FASTA manifest"
    )
    base_name = os.path.splitext(os.path.basename(os.fspath(INPUT_BLAST_TABULAR)))[0]
    OUTPUT_HDF5 = os.path.join(
        NETWORK_DIR,
        f"{base_name}_[BLAST]_EValue.h5",
    )


def main(argv=None):
    load_tool_settings(globals(), __file__, PROJECT_ROOT, argv)
    print("--- Converting External BLAST Results to HDF5 ---")
    try:
        configure_runtime_paths()
        summary = build_blast_network(
            FULL_INPUT_BLAST_TABULAR,
            FULL_INPUT_FASTA,
            OUTPUT_HDF5,
            layout=BLAST_LAYOUT,
            query_column=QUERY_COLUMN,
            subject_column=SUBJECT_COLUMN,
            evalue_column=EVALUE_COLUMN,
            matrix="Imported",
            batch_size=1000000,
            show_progress=True,
        )
    except (BlastParseError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print("=" * 40)
    print("PARSING DIAGNOSTICS")
    print("=" * 40)
    print(f"FASTA Headers:        {summary.fasta_header_count}")
    print(f"BLAST Data Rows:      {summary.data_rows}")
    print(f"Self Rows Ignored:    {summary.self_rows}")
    print(f"Unique Edges Saved:   {summary.unique_edges}")
    print(f"Output:               {summary.output_path}")
    print("=" * 40)
    print("Conversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
