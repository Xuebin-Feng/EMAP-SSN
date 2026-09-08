# MCP settings (server 0.9.0, pipeline settings schema 1)

## Three workflow tools (breaking migration)

Only `emapssn_pipeline`, `emapssn_viewer_data`, and `emapssn_viewer_control`
are registered. Old MCP tool names are removed, with no aliases or legacy mode.
Restart the server and refresh the client's tool catalog; update saved calls.
Host approval persistence is controlled by the MCP client, not this server.

Call `help` inside each entry point for its action catalog, then `describe` for
an individual action's argument schema, effects and example. Arguments are strict
and reject unknown keys. Example MCP tool name: `emapssn_pipeline`; call arguments:

```json
{"action": "describe", "arguments": {"action": "start_layout_job"}}
```

```json
{"action": "start_layout_job", "arguments": {"settings_path": "layout.json"}}
```

Layout generation remains separate from pipeline IDs but shares their job queue.
The cache workflow is pipeline `export_layout_settings` → `start_layout_job` →
`get_job` / `read_log` → `inspect_file`, followed by viewer-control
`export_settings` → `validate_settings` → `start_session`.
The two export actions do not accept `kind`. Existing action result payloads,
node limits, file contracts, queue ownership and connection isolation are preserved.
Viewer-data is read-only; the other entry points can write files or change state.
Richer viewer analyses and response-size budgets are outside this migration.

Below, notation such as `emapssn_pipeline(action="inspect_file")` identifies the
tool and selected action. Unless a snippet includes `action`, its JSON is the
contents of the `arguments` object, not the complete MCP call.


## Agent navigation instructions

For a new Viewer use `emapssn_viewer_control(action="export_settings")`; for a
layout use `emapssn_pipeline(action="export_layout_settings")`. Export merges saved `viewer_settings.json`
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
there is no embedded fallback. The 0.8.0 migration replaces the old public tool names; backend execution
contracts remain unchanged.

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

`emapssn_viewer_data(action="list_sessions")` pages compact session identities and
input paths without allocating snapshots. `get_summary` captures immutable backend
metadata/memberships and returns compact cache provenance (status, filename,
manifest ID), not complete generation documents. For full file provenance use
pipeline action `inspect_file` on the known cache path. Provenance checks read no
numerical datasets and do not establish numerical correctness.

### Snapshot data access (0.9.0 migration)

The public MCP catalog still contains exactly three entry points. All following
actions belong to `emapssn_viewer_data`, using its `action` and `arguments` shape:

1. `get_summary` captures a fresh snapshot and returns `snapshot_id`, capture time,
   population counts, metadata preview, loaded edge count, alignment coverage and
   effective reference information. Existing snapshots do not follow Viewer edits.
2. `describe_fields` pages field types, valid/missing/invalid counts and provenance
   availability. Historical per-column source information is unavailable.
3. `create_subset` requires `snapshot_id` and `scope` (`all`, `visible`, `selected`),
   with an optional Boolean selection `expression`. Supported atoms are headers,
   metadata, labels and `$sele$`; file and residue atoms are rejected before evaluation.
4. `summarize_subset` accepts optional `subset_id` and `columns`; omitted subset
   means the whole snapshot. Numeric `quantiles` are min, Q1, median, Q3, max.
   Text output includes top ten categories, other/missing counts, and pageable full
   category counts. Membership rows include noise separately; groups can overlap.
5. `query_nodes` requires `snapshot_id`, accepts optional `subset_id` and explicit
   `columns`, and returns `rows`. Default limit is 25; omitted columns means no
   metadata. This intentionally replaces the old live scope/offset query contract.
6. `read_value` takes snapshot ID, node `index`, `field` (`node_id`, `groups`, or
   `metadata`) and metadata `column` when applicable. Concatenate returned `text`
   slices using `next_offset`, then JSON-decode once to recover the exact value.
   Group-category references also supply `member_index` to identify one label
   unambiguously; omit it when retrieving the node's complete membership list.

All snapshot data responses have a default serialized UTF-8 JSON budget of 16384
bytes, configurable with `max_bytes` from 1024 to 65536. Pagination uses opaque
`next_cursor` values bound to the snapshot, subset, action and query. Changing the
page limit or budget is allowed; changing columns is a different query. Oversized
node values carry explicit omission markers and exact retrieval references.
If one indivisible row cannot fit, request fewer columns or a larger budget.
`list_sessions` uses row offsets (`next_offset`) and reflects changing live discovery;
`read_log` retains byte offsets and now defaults to 8192 bytes.

Snapshots live only in the Viewer process: two snapshots, 256 MiB conservative
accounted storage, 15-minute idle expiry, 32 subsets per snapshot. Capture rejects
oversized datasets before copying and evicts least-recently-used idle snapshots.
Expired or cross-Viewer IDs produce refresh-required errors. There are no file
writes, GUI selection changes or command executions. Older Viewers must be restarted
after upgrade; the client checks `snapshots_v1` capability before sending data requests.

The Qt bridge copies only the required source data. Filtering, aggregation,
provenance reads and formatting run in HTTP workers, outside the Qt event thread.
Sequence/edge records, MSA matrices, coordinates and command-derived analysis are
outside this stage. Numeric NaN/None/blank values count as missing; infinities and
unparseable numeric values count as invalid. Non-finite record values are explicitly
tagged as {"nonfinite":"NaN"}, {"nonfinite":"Infinity"}, or {"nonfinite":"-Infinity"}
rather than silently converted to null. Unknown annotation sources are not inferred.
Selection expressions retain the command engine grammar, including `$sele$` and
its restrictions on metadata property names. JSON column projection and read_value
preserve arbitrary column names, including commas and Unicode.


## Export, edit, validate, execute

Agents can export saved application settings instead of constructing a complete
JSON document. `emapssn_pipeline(action="export_tool_settings")(tool_id, output_path=None)` exports only
the selected tool and its required directories, inheriting `tools_settings.json`
and filling missing values from the tool defaults. Empty selectors remain editable.

The settings-export actions accept `output_path` and `settings_path`:

- `emapssn_pipeline(action="export_layout_settings")`: generation inputs, filtering, physics/UMAP settings and output
  destination. Visual settings and MSA display settings are excluded.
- `emapssn_viewer_control(action="export_settings")`: a full effective snapshot of Inputs, Visuals, Physics and
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
with `emapssn_pipeline(action="validate_settings")` or Viewer JSON with `emapssn_viewer_control(action="validate_settings")`,
and submit its path to `emapssn_pipeline(action="start_job")`, `emapssn_pipeline(action="start_layout_job")` or
`emapssn_viewer_control(action="start_session")`. Layout settings are validated on submission and cache/input
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

Use `emapssn_pipeline(action="list_tools")` to choose one of the 14 pipeline IDs, then
`emapssn_pipeline(action="get_tool_schema")` with `{"tool_id": "sanitize_sequences"}` to discover
accepted parameters, descriptions, native JSON types, defaults, allowed choices,
conditional inputs, directory semantics, and an example. Discovery reads static
metadata without importing models or initializing a GUI or accelerator.

## Hardware discovery

Call `emapssn_pipeline(action="get_compute_capabilities")` with `{}` for runtime devices and system resources,
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

Use `emapssn_pipeline(action="inspect_file")` before submission when file-level evidence is useful:

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

Call `emapssn_pipeline(action="validate_settings")` with:

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
`emapssn_pipeline(action="start_job")`. The job snapshots the same normalized document before
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

After exporting and editing layout JSON, call `emapssn_pipeline(action="start_layout_job")` with its path:

```json
{
  "settings_path": "layout.json"
}
```

Layout jobs are enqueued into the same unified FIFO queue managed by `PipelineJobManager`
and share job tracking, status monitoring, log streaming (`emapssn_pipeline(action="read_log")`), and
cancellation (`emapssn_pipeline(action="cancel_job")`).

The worker routes through headless Config to `Layout_Cache_Generator.py`. It
publishes an HDF5 coordinate cache, a compatible-folder manifest and a canonical
sanitized FASTA backup under `SAVED_LAYOUT_DIR`. Use the actual cache path in the
completed job's `output_locations` rather than reconstructing it from the preview.
The existing individual-parameter and `settings_document` forms remain supported;
omitting `cache_filename` in the individual-parameter form now selects automatic naming.

## Viewer sessions

Viewer sessions are independent of MCP connections. Use `emapssn_viewer_data(action="list_sessions")`
to discover authenticated Viewers started through the GUI, CLI, or MCP.
`emapssn_viewer_control(action="connect_session")` selects an existing session without changing its settings.
Supply `session_id`, or omit it only when exactly one Viewer is running.

`emapssn_viewer_data(action="get_summary")`, `emapssn_viewer_data(action="query_nodes")`, and `emapssn_viewer_control(action="close_session")` use the
connected session when `session_id` is omitted. An explicit ID affects only that
operation. Without a connection or explicit ID, they return a selection error.
Each MCP connection has its own selection; switching sessions leaves the previous
Viewer running.

`emapssn_viewer_control(action="disconnect_session")` clears the selection and leaves the Viewer running.
It is safe to repeat. Closing the MCP connection or server also leaves ready
Viewers running. Use `emapssn_viewer_control(action="close_session")` explicitly to terminate a Viewer;
success is returned only after process exit is verified. Failed closure retains
session information. A disconnected client never automatically reconnects.

### Launch with complete JSON

`emapssn_viewer_control(action="start_session")` accepts exactly one of `settings_document` or
`settings_path`, plus `mode` (`normal` or `headless`). The old standalone
`cache_path` argument is no longer accepted. A successful launch also connects
the caller to the new session. Logs are written to the returned `stdout_log` and
`stderr_log` paths, separately from MCP protocol output.

Normal MCP launches open a visible terminal alongside the Viewer. Both output
streams are copied to that terminal and retained in the launch logs, including
native-library output and partial progress lines. Headless launches retain the
same logs without opening a terminal. `emapssn_viewer_data(action="read_log")` pages either stream using
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

Use `emapssn_viewer_control(action="get_settings_schema")` to discover fields and defaults, and
`emapssn_viewer_control(action="validate_settings")` with the same settings source to validate files and
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
`emapssn_viewer_control(action="connect_session")`; connect/disconnect does not require launching a process.
