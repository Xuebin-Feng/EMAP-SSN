# MCP settings (server 0.7.0, pipeline settings schema 1)

## Agent navigation instructions

For a new Viewer or layout, agents must start with `export_config_settings`
(`kind="viewer"` or `kind="layout"`). Export merges saved `viewer_settings.json`
preferences into the output, including visual, simulation, and physics values.
Edit only requested fields and necessary dependent inputs, then validate/execute
the full exported JSON. A Viewer cache-selection overlay can be supplied through
the export tool's `settings_path`. Building a minimal document from schema defaults
bypasses saved preferences; validation and execution intentionally consume the
explicit snapshot instead of rereading personal settings.

The canonical workflow guide is
[`Agent_Instructions.md`](../src/mcp_server/Agent_Instructions.md). The server
loads this UTF-8 file relative to its own module and transmits its exact contents
in the MCP initialization `instructions` field. Agents do not need repository
access to receive it; whether instructions are surfaced to the model is controlled
by the MCP host. Existing tool descriptions also explain prerequisites and next
steps for clients navigating through tool discovery.

Maintain workflow guidance in that file and parameter rules in tool schemas.
Restart the MCP server after changing the guide or tool descriptions. A missing,
unreadable, invalid UTF-8, or empty guide causes an actionable startup error;
there is no embedded fallback. These documentation changes add no tools and do
not change execution contracts.

## Viewer aliases and cache provenance

Viewer titles include an eight-character UUID alias, such as
`EMAP-SSN Viewer [A7C92F10]`. Identity belongs to the Viewer instance and survives
internal web-server restarts. A newly opened Viewer gets a new UUID even when it
loads the same cache. Qt sets the native window title on Windows, Linux and macOS;
whether native decorations are visible is controlled by the desktop/window manager.

MCP session arguments accept either a full UUID or the exact eight-character alias
(case-insensitive). Ambiguous aliases fail with the matching full UUIDs. Connections
continue to store full UUIDs internally. Listings, launch results and connect results
expose `session_alias` alongside the existing `session_id`.

`list_viewer_sessions` and `get_viewer_summary` expose `cache_metadata`, containing
the absolute `cache_path`, `cache_filename`, the three provenance `attributes`,
parsed `generation_parameters`, and the parent `folder_manifest`. The reader checks
the manifest's own identity, its match to `cache_manifest_id`, and the SHA-256
`layout_compatibility_id` against the canonical parameter JSON. It reads no datasets
and runs outside the Qt UI thread. Results are refreshed when file identity, size
or timestamps change and do not rewrite caches.

Metadata `status` is `complete`, `partial`, `invalid`, or `unavailable`, with
`diagnostics`. Older caches missing provenance remain discoverable. If a Viewer
stops responding during listing, its entry retains identity and reports
`metadata_error`. Generation parameters describe the saved cache, while fields such
as `active_threshold` describe current interactive state. A valid provenance hash
does not establish that coordinates have never been manually edited.

## Export, edit, validate, execute

Agents can export saved application settings instead of constructing a complete
JSON document. `export_pipeline_settings(tool_id, output_path=None)` exports only
the selected tool and its required directories, inheriting `tools_settings.json`
and filling missing values from the tool defaults. Empty selectors remain editable.

`export_config_settings(kind, output_path=None, settings_path=None)` supports:

- `kind="layout"`: generation inputs, filtering, physics/UMAP settings and output
  destination. Visual settings and MSA display settings are excluded.
- `kind="viewer"`: a full effective snapshot of Inputs, Visuals, Physics and
  Directories, with the selected existing cache. This is configuration, not window state.

Config inherits saved `viewer_settings.json`. An optional edited JSON file overlays
these preferences; an existing layout document can also be re-exported as layout
JSON to refresh its destination. Config export needs valid input files and enough
settings to determine compatibility. It never invents missing scientific inputs.
Unsaved GUI edits and named profiles are not read by headless export. GUI exports
continue to use current controls and share the serialization helpers.

Exports contain absolute directory paths, resolving project-relative paths and
Config directory aliases. Omitted output paths create uniquely named JSON files
in the saved `SETTING_EXPORT_DIR`; explicit output paths must not exist. Export
does not update personal settings, compute a layout or reserve a cache filename.

Each result contains `settings_path` and `settings_document`; Config results also
contain `cache_path` and `cache_filename`. Edit the file, validate pipeline JSON
with `validate_pipeline_settings` or Viewer JSON with `validate_viewer_settings`,
and submit its path to `start_pipeline_job`, `start_layout_job` or
`start_viewer_session`. Layout settings are validated on submission and cache/input
compatibility is rechecked in the worker. Execution does not reapply saved preferences.

### Cache naming and Viewer selection

Layout JSON includes `CACHE_FILENAME`, `TARGET_CACHE_PATH` and `CACHE_NAME_MODE`:

- `auto` exports the next `version_NN.h5` and its absolute path as a preview.
  Execution resolves the destination again using the edited inputs. If occupied,
  it advances to the next version and never overwrites a cache.
- `explicit` honors the filename and optional path, rejecting mismatches,
  incompatible folders and occupied files. To request a custom name, switch to
  `explicit` and update both the filename and path. Older documents without
  `CACHE_NAME_MODE` retain explicit-name behavior.

MCP layout jobs execute `EMAPSSN_Config.py --headless generate-layout`, which calls
the existing layout generator. Writers using the same layout root are serialized
with an OS lock through generation/publication, including across MCP processes;
different layout roots remain independent. Process termination releases the lock.
Atomic no-overwrite publication also handles a competing external file writer.
The completed job's `output_locations` includes `TARGET_CACHE_PATH`,
`CACHE_FILENAME`, and `EXECUTED_SETTINGS` (a separate effective settings snapshot).
The original submitted snapshot is retained unchanged. Generation alone opens no Viewer.

Viewer export selects the most recently modified cache in the unique compatible
folder; filename ascending breaks modification-time ties. No cache or multiple
compatible folders is an error. A `TARGET_CACHE_PATH` in the overlay selects a
specific cache. When opening a newly generated layout, supply the actual path
returned by its job. Viewer JSON includes `CACHE_FILENAME`, which must match that
path; older Viewer JSON may omit it. Launch validates and opens exactly that cache,
without silently generating or selecting a different one.

MCP Viewer launch runs detached headless Config, which dispatches into Viewer in
the same process to preserve PID tracking on Windows and Unix. Existing readiness,
private logs, connection and shutdown handling remain in MCP. `normal` opens a
visible Viewer; `headless` uses offscreen Qt. Neither opens the Config window.

### CLI equivalents

From the project root, using the managed Python interpreter:

```text
python src/EMAPSSN_Tools.py --headless export --tool sanitize_sequences
python src/EMAPSSN_Config.py --headless export --kind layout --output layout.json
python src/EMAPSSN_Config.py --headless export --kind layout --settings layout.json
python src/EMAPSSN_Config.py --headless generate-layout --settings layout.json
python src/EMAPSSN_Config.py --headless export --kind viewer --output viewer.json
python src/EMAPSSN_Config.py --headless launch-viewer --settings viewer.json --viewer-mode normal
```

The example filenames are chosen destinations, not pre-existing files. CLI export
writes a JSON result to stdout; diagnostics go to stderr. Runtime calculation logs
remain in the MCP job's private log files.

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
`alignment_fasta`, `embedding`, `network`, `sparse_msa`, `blast_tabular`,
`layout_cache`, and `settings`. Detection uses contents and HDF5 structure, not the
extension. Aligned FASTA should be specified explicitly; `tool_id: "sparse_msa_converter"`
also selects aligned-FASTA checks for auto-detected FASTA. Layout caches can be detected
automatically or inspected with `file_type: "layout_cache"`.
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

## Layout cache generation

After exporting and editing layout JSON, call `start_layout_job` with its path:

```json
{
  "settings_path": "layout.json"
}
```

Layout jobs are enqueued into the same unified FIFO queue managed by `PipelineJobManager`
and share job tracking, status monitoring, log streaming (`read_pipeline_log`), and
cancellation (`cancel_pipeline_job`).

The worker routes through headless Config to `Layout_Cache_Generator.py`. It
publishes an HDF5 coordinate cache, a compatible-folder manifest and a canonical
sanitized FASTA backup under `SAVED_LAYOUT_DIR`. Use the actual cache path in the
completed job's `output_locations` rather than reconstructing it from the preview.
The existing individual-parameter and `settings_document` forms remain supported;
omitting `cache_filename` in the individual-parameter form now selects automatic naming.

## Viewer sessions

Viewer sessions are independent of MCP connections. Use `list_viewer_sessions`
to discover authenticated Viewers started through the GUI, CLI, or MCP.
`connect_viewer_session` selects an existing session without changing its settings.
Supply `session_id`, or omit it only when exactly one Viewer is running.

`get_viewer_summary`, `query_viewer_nodes`, and `close_viewer_session` use the
connected session when `session_id` is omitted. An explicit ID affects only that
operation. Without a connection or explicit ID, they return a selection error.
Each MCP connection has its own selection; switching sessions leaves the previous
Viewer running.

`disconnect_viewer_session` clears the selection and leaves the Viewer running.
It is safe to repeat. Closing the MCP connection or server also leaves ready
Viewers running. Use `close_viewer_session` explicitly to terminate a Viewer;
success is returned only after process exit is verified. Failed closure retains
session information. A disconnected client never automatically reconnects.

### Launch with complete JSON

`start_viewer_session` accepts exactly one of `settings_document` or
`settings_path`, plus `mode` (`normal` or `headless`). The old standalone
`cache_path` argument is no longer accepted. A successful launch also connects
the caller to the new session. Logs are written to the returned `stdout_log` and
`stderr_log` paths, separately from MCP protocol output.

Normal MCP launches open a visible terminal alongside the Viewer. Both output
streams are copied to that terminal and retained in the launch logs, including
native-library output and partial progress lines. Headless launches retain the
same logs without opening a terminal. `read_viewer_log` pages either stream using
byte offsets (`stream`, `offset`, and `limit`); use the full session ID to read
retained output after disconnecting or closing within the same MCP transport.

Closing the Viewer window clears its selected connection automatically (after
three failed discovery checks, about three seconds). Listing sessions or querying
a missing selected session clears the selection immediately. Backend transport
shutdown also clears the connection and stops its monitor; the independently
running Viewer is left available for a later connection.

```json
{
  "mode": "headless",
  "settings_document": {
    "TARGET_CACHE_PATH": "my_layout/version_00.h5",
    "NODE_FASTA_FILE": "$input_file$/Sequence_Sets/proteins.fasta",
    "INPUT_HDF5": "$input_file$/Networks_EValues/proteins_network.h5",
    "MSA_FILE": "",
    "ALIGNMENT_REFERENCE": "",
    "ALIGNMENT_SCORE": "global",
    "NORM_MODE": "alignment_length",
    "UMAP_MODE": false,
    "SIMILARITY_THRESHOLD": 0.1,
    "TOP_EDGE_PERCENT": null,
    "NODE_SIZE": 10
  }
}
```

Use `get_viewer_settings_schema` to discover fields and defaults, and
`validate_viewer_settings` with the same settings source to validate files and
return the normalized document before launching. Required scientific settings
must match the cache manifest. Alignment networks require score and normalization
settings. Physics layouts require explicit threshold and top-percent fields;
set the inactive filter to null. UMAP requires `UMAP_NEIGHBORS` and null edge
filters. `MSA_FILE` must be explicit; an empty string disables alignment loading.
A supplied alignment reference must exist in the selected MSA.

Documents use the existing flat Viewer settings format. Directory aliases
`$input_file$`, `$cache_file$`, and `$analysis_result$` resolve from the supplied
base directories or documented project-relative defaults. Bare input filenames
resolve under their configured input directory; input paths containing directories
resolve against the project root. Relative `TARGET_CACHE_PATH` values resolve
under `SAVED_LAYOUT_DIR`. Absolute paths are accepted. A settings file's location
does not change these rules.

Explicit documents never inherit personal `viewer_settings.json` values. Missing
files, mismatching fingerprints, incompatible scientific parameters, and cache
header-order mismatches are errors. The Viewer never searches for substitute
files or silently drops connectivity. Presentation defaults are included in the
normalized snapshot. The minimal cache format is unchanged.

For direct CLI use:

```text
python src/EMAPSSN_Viewer.py --settings viewer_launch.json --headless
```

Caller-owned JSON files are preserved. GUI and MCP launchers consume their own
private snapshots after successful validation. Failed/cancelled starts clean up
only their own process trees and retain diagnostic logs.

On Windows, independent launch requires the host to permit Job Object breakaway.
A host that forbids it receives a startup error rather than a session that dies on
MCP disconnect. In that host, open the Viewer through the GUI or CLI and use
`connect_viewer_session`; connect/disconnect does not require launching a process.
