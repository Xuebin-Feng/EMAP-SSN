# MCP pipeline settings (server 0.4.0, settings schema 1)

Use `list_pipeline_tools` to choose one of the 14 pipeline IDs, then
`get_pipeline_tool_schema` with `{"tool_id": "sanitize_sequences"}` to discover
accepted parameters, descriptions, native JSON types, defaults, allowed choices,
conditional inputs, directory semantics, and an example. Discovery reads static
metadata without importing models or initializing a GUI or accelerator.

## Hardware discovery

Call `get_compute_capabilities` with `{}` for runtime devices and system resources,
or `{"tool_id": "align_similarity_matrix"}` to include applicable pipeline settings.
This read-only endpoint enumerates devices usable by the server's Python environment,
preserving installer-approved filtering. It does not scan installed hardware that
the runtime cannot use. ROCm is identified separately from CUDA, while both retain
the PyTorch `cuda:N` device identifier.

The result includes a timestamp, Python/PyTorch/runtime versions, CPU counts,
system memory, device memory, and capability evidence. Status is `ok`, `partial`,
or `unavailable`; failed collection must not be interpreted as CPU-only success.
Unavailable measurements are null and explained by the associated `errors` or
memory reason. Apple unified working-set values are not dedicated VRAM/free memory.
Memory values are observations in bytes, not reservations for future jobs.

Capabilities are `supported`, `unsupported`, or `unknown`, with their source and
reason. Every capability is explicitly unverified by computation. NVIDIA compute
capability supplies TF32/BF16 hardware eligibility; backends without a suitable
metadata check report BF16 as unknown. Tiled support checks API presence, not
successful execution or sufficient memory. Tool-specific restrictions are reported
separately (for example, network injection cannot use MPS tiled execution).

Collection uses a short-lived helper with the server's interpreter and a 20-second
timeout. It may initialize the helper's runtime through metadata queries, but does
not allocate tensors, benchmark, synchronize, or clear accelerator caches. Timeout
terminates only this helper, leaving pipeline jobs untouched. No fastest-device
recommendation is made: `auto` still selects at execution time. Model and source-file
constraints remain runtime checks. Settings validation is unchanged.

## Inspect an explicitly selected file

Use `inspect_pipeline_file` before submission when file-level evidence is useful:

```json
{"path": "Embeddings/proteins_embeddings.h5", "tool_id": "embedding_msa"}
```

Paths may be absolute (including outside the project), or relative to the project
root. No directory inventory is performed. The helper opens files read-only and
does not create jobs, fix files, or change submission requirements.

`file_type` defaults to `auto`. Supported explicit types are `fasta`,
`alignment_fasta`, `embedding`, `network`, `sparse_msa`, `blast_tabular`, and
`settings`. Detection uses contents and HDF5 structure, not the extension. Aligned
FASTA should be specified explicitly; `tool_id: "sparse_msa_converter"` also
selects aligned-FASTA checks for auto-detected FASTA. Layout caches are excluded.
Ambiguous plain tabular files require an explicit format:

```json
{
  "path": "external/results.tsv",
  "file_type": "blast_tabular",
  "tool_id": "parse_blast_output",
  "parameters": {
    "BLAST_LAYOUT": "custom_columns",
    "QUERY_COLUMN": 1,
    "SUBJECT_COLUMN": 2,
    "EVALUE_COLUMN": 3
  }
}
```

`parameters` requires a `tool_id`, uses that tool's strict keys/types, and does not
require unrelated job inputs. BLAST defaults to `standard_outfmt6`; specify
`outfmt7_fields` to validate Query/Fields comment context. Supplying a tool ID
does not establish cross-file compatibility or guarantee that all job inputs exist.

The report separates:

- `structural_validity`: `valid`, `invalid`, or `unknown`, limited to performed checks.
- `generation_completion`: `complete`, `incomplete`, or `unknown`, with evidence.
- `network_pair_coverage`: `all-pairs count reached`, `sparse`, or `unknown`, with
  observed/expected counts. Unique pairs are never claimed verified from counts.

Embedding inspection checks completion flags, required metadata, sanitized
header/sequence manifests, and each embedding's presence, shape, and dtype. It
does not read embedding arrays. Network inspection checks type-specific datasets,
ranks, dtypes, and counts without reading edges or scores. Sparse MSA inspection
checks CSR dimensions and mapping metadata without scanning CSR arrays or
reconstructing the alignment.

A valid sparse network is not automatically unfinished. Selected-edge counts can
be inherited by network extraction, so count differences are reported as warnings,
not standalone proof of interrupted generation. Retained alignment `_resume`
state indicates unfinished publication. Explicit completion flags are reported as
evidence, but malformed structures prevent confirmation of a complete result.
Without a trustworthy completion marker, generation status stays unknown—even
for a final-looking filename or a structurally valid file.

FASTA/BLAST inspection streams text and checks record/row structure. Aligned FASTA
also checks equal lengths. Lowercase or unusual FASTA characters are reported as
sanitization warnings, not silently changed. BLAST numeric values and cross-file
header matching are not checked. Settings JSON uses strict MCP validation when a
tool is supplied; otherwise only recognized sections and outer structure are
checked. Reports do not contain full sequences, arrays, or header lists.

Inspection is bounded by a 20-second helper timeout, an internal 18-second scan
budget, 100,000 enumerated metadata records per checked collection, 8 MiB JSON
metadata, and 1 MiB text lines. At most 50 findings are returned; omitted findings
are counted. Reaching a limit or a read failure leaves inspection unfinished and
cannot confirm completion. Reports list performed and omitted checks plus examined
record/line counts. File size, modification time, and identity are compared before
and after; detected changes make the result unstable and inconclusive. These
checks cannot detect every possible concurrent write, so inspect quiescent files.

Structural inspection does not validate numerical values, edge ordering or
uniqueness, CSR pointer values, or future runtime success. It remains optional:
queued jobs may depend on files that preceding jobs have not yet produced.

## Preview and submit

Call `validate_pipeline_settings` with:

```json
{
  "tool_id": "sanitize_sequences",
  "parameters": {
    "INPUT_FASTA": "proteins.fasta",
    "ENABLE_LENGTH_FILTER": true,
    "MIN_SEQ_LENGTH": 50,
    "MAX_SEQ_LENGTH": 1000,
    "OVER_WRITE": false
  },
  "directories": {"FASTA_DIR": "Input_Files/Sequence_Sets"}
}
```

The preview returns `valid`, field-specific `errors`, a normalized
`settings_document`, `effective_directories`, `applied_defaults`, and `overrides`.
It creates no files or jobs. If valid, send the same arguments to
`start_pipeline_job`. The job snapshots the same normalized document before
execution. Invalid submissions do not create job files or enter the queue.

Defaults come from the published contracts, never the GUI's last-used settings.
Relative directories resolve against the server's project root. Omitted, blank,
or null directories use project defaults. Input filenames retain the individual
tool's directory semantics; discovery does not assert that input files exist.

Exactly one of `parameters`, `settings_document`, or `settings_path` is required.
An empty `parameters` object counts as a supplied form. `directories` can only
accompany `parameters` and accepts that tool's directory keys.

## Existing exports

Both preview and submission also accept an exported file:

```json
{
  "tool_id": "align_similarity_matrix",
  "settings_path": "Cache_Files/Exported_Settings/alignment.json"
}
```

Alternatively supply the document directly:

```json
{
  "tool_id": "align_similarity_matrix",
  "settings_document": {
    "DIRECTORIES": {},
    "Align_Similarity_Matrix.py": {
      "INPUT_HDF5": "proteins_embeddings.h5",
      "BATCH_SIZE": 500000,
      "DEVICE_SELECTION": "auto"
    }
  }
}
```

Documents require `DIRECTORIES` and the selected script section. Recognized
unrelated tool sections and global directories are allowed in full GUI exports;
only the selected tool's settings are normalized. Unknown section names,
parameter names, and directory keys are rejected.

All three MCP forms use strict types: `500000` and `false` are valid native
values; `"500000"` and `"false"` are not. Numeric settings reject booleans and
non-finite numbers. `NORM_THRESHOLD` uses JSON `null` to disable filtering.
`HOST_CACHE_GB` accepts a non-negative number or the literal string `"auto"`.
Strings such as headers, highlight expressions, and paths remain strings.

GUI exports now convert numeric text controls before writing. Export errors
identify invalid fields instead of silently retaining unparseable strings.
Incomplete configurations can still be exported, but MCP submission enforces
required inputs. Existing files are not rewritten: re-export or correct rejected
older files using the reported fields. Direct CLI and GUI execution retain their
existing settings-loading behavior.

## Validation and headless behavior

The schemas describe static constraints and conditional inputs, including MSA's
optional FASTA filter and manual versus stored PWA/search sequences. Additional
cross-field checks require ordered enabled sequence-length bounds, distinct custom
BLAST columns, accelerator selection for tiled execution, and CUDA or automatic
selection for explicit TF32. Manual sequences must survive sanitization.

Validation does not load models, contact services, inspect input file contents,
check credentials, or benchmark hardware. A successful preview establishes a valid
configuration, not runtime readiness. Inputs may be outputs of preceding queued
jobs. Model choices come from installed static plugin metadata; device selection
uses `auto`, `cpu`, `mps`, `cuda:N`, or `xpu:N`, with availability checked at runtime.

MCP MSA jobs force `SHOW_REGRESSION_PLOT` to `false` after validating its supplied
type and report any override. GUI exports preserve the user's plotting preference.
Headless sanitization prints the same 50-bin length distribution as a text table
without creating a figure. Queue ownership, cancellation, subprocess logging, and
output-directory reporting are unchanged.
