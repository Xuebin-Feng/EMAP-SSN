# EMAP-SSN agent guide

Use this server to prepare sequence data, run supported scientific pipelines,
calculate network layouts, and launch or inspect local Viewers. Choose the entry
point matching the user's task; do not run every discovery tool for every request.

| Workflow entry point | Responsibilities |
| --- | --- |
| `emapssn_pipeline` | Pipeline discovery, preparation, execution, layout-cache generation and shared job monitoring |
| `emapssn_viewer_data` | Read sessions, summaries, node pages and captured Viewer logs |
| `emapssn_viewer_control` | Export/validate Viewer settings, launch, connect, disconnect and close |

Each call supplies an action and an arguments object. Use action="help" with no
arguments to list actions. Use action="describe", arguments={"action": "start_job"}
on the pipeline entry point to obtain that action's schema, effects and example.
Put all operation parameters inside arguments. Discovery does not execute work.
Unknown actions, unexpected arguments and invalid types fail before dispatch.

## Run a pipeline

Start with `emapssn_pipeline(action="list_tools")` when the appropriate pipeline is unknown. Its
entries contain stable `tool_id` values, descriptions, and directory contracts.
A pipeline ID is an argument to MCP tools, not itself an MCP tool name. Pass the
chosen ID to `emapssn_pipeline(action="get_tool_schema")` for accepted parameters, defaults,
conditions, and an example. Do not invent IDs, script names, or parameter keys.

Use `emapssn_pipeline(action="inspect_file")` on known input paths when their contents or format
need checking. It inspects a selected file; it does not search directories for
inputs. Use `emapssn_pipeline(action="get_compute_capabilities")` when device availability affects a choice,
optionally supplying the pipeline ID. This reports runtime metadata without
benchmarking or executing models; it does not establish that a computation will
fit in memory or succeed.

Prepare exactly one input form: `parameters`, `settings_document`, or
`settings_path`. Optional `directories` accompanies `parameters` only. Use native
JSON numbers and booleans. Call `emapssn_pipeline(action="validate_settings")`, examine `valid` and
field-specific `errors`, and correct invalid settings before submission. A valid
preview supplies a normalized `settings_document` suitable for
`emapssn_pipeline(action="start_job")`. Configuration validation does not check every input file,
credential, or hardware requirement.

Submit the chosen pipeline with `emapssn_pipeline(action="start_job")`, save its `job_id`, and
follow it with `emapssn_pipeline(action="get_job")`. Submission means queued or running, not
completed. Settings can cause output files to be created or overwritten; align
those choices with the user's request.

## Reuse saved settings

For Viewer launches use viewer-control action="export_settings"; for layouts use
pipeline action="export_layout_settings". Both inherit the relevant `viewer_settings.json` preferences: layout exports
include simulation/physics; Viewer exports include alignment/display preferences. Execute the full exported JSON with
only necessary edits. Never construct minimal JSON from schema defaults instead.
Use `emapssn_pipeline(action="export_tool_settings")` for saved pipeline settings. Exports create files;
explicit output paths must be unused, while omitted paths are allocated automatically.

When editing JSON, preserve user-defined parameters unless they invalidate the
job. For a different input file, obtain the existing JSON and replace only that
field, plus necessary dependent fields. For example, paired embedding and network
files may need matching sources. Validate compatibility; do not reset unrelated
settings to defaults or guess dependent files. Explain any required changes.

When reporting calculations or generations, always list the effective parameters
and input files actually used, including preserved values and applied defaults,
from the executed settings rather than just the requested edits.

Review the exported settings, change the intended fields, then validate and
execute using the resulting document or path. Exporting does not execute a job.
An exported layout cache filename is a preview, not a reservation. Viewer exports
may select the newest compatible cache: verify that it is the user's intended
cache. Supplied execution documents are not silently refreshed from personal
settings. Do not assume a settings file's directory is the base for relative
input paths; follow the relevant tool's path and directory rules.

## Calculate a layout

First call `emapssn_pipeline(action="export_layout_settings")`. Preserve the exported
simulation and physics values (including forces, timestep, convergence, step
limit, and packing) unless changes are required. Pass the edited document/path to
`emapssn_pipeline(action="start_layout_job")`; individual arguments can substitute built-in defaults. Layout
JSON differs from pipeline parameters and the sectioned Viewer document.

Provide the node FASTA and network HDF5, choose physics or UMAP, and use scientific
settings consistent with the network. Ask for missing scientific choices when
they cannot be established from the user's request or inspected metadata. The
start operation validates before enqueueing; there is no separate layout
validation tool. Resolve export errors instead of bypassing saved preferences.
Track the returned job using the shared job tools. After success, inspect the
actual output cache and use its identity and matching inputs to prepare Viewer
settings. Do not assume a layout job opens a Viewer.

## Open and inspect a Viewer

When the user means an existing Viewer, call `emapssn_viewer_data(action="list_sessions")` first. Match
the intended session using its identity and cache metadata, then connect. Listing
does not select a session. With multiple candidates, resolve the intended target
rather than arbitrarily choosing one.

For a new Viewer, first call `emapssn_viewer_control(action="export_settings")`. To select
another cache, its optional `settings_path` can supply an overlay with
inputs.TARGET_CACHE_PATH and necessary matching inputs. Overlays require
schema_version=2 and kind="viewer", with edits inside the named sections. Keep the full export, consult
`emapssn_viewer_control(action="get_settings_schema")`, then call `emapssn_viewer_control(action="validate_settings")` and pass its
normalized document to `emapssn_viewer_control(action="start_session")`. Validation checks source files and
cache compatibility. Inheritance happens during export; validation must not replace
preserved preferences with a minimal payload. alignment.MSA_FILE="" disables alignment.

Both execution formats require schema_version=2 and the matching kind. Layout
sections are inputs, network, layout, simulation, physics, packing, and output.
Viewer sections are inputs, alignment, visualization, and directories. Legacy
execution JSON must be re-exported; personal settings stay unchanged. Never add
network, layout, physics, BOX_SCALE, or cache_fallback to Viewer JSON: verified
cache provenance supplies score mode, normalization, edge filters, UMAP settings,
and box scale internally. Input FASTA/network paths remain explicit because cache
metadata records fingerprints and basenames, not source locations. Missing or
inconsistent provenance is an error, not permission to invent defaults. Report
cache-derived settings from inspection provenance separately from Viewer JSON.

Use the default `normal` mode for a visible Viewer and terminal. Choose `headless`
only when a desktop window is not wanted. Successful launch returns a ready
session and connects this transport to it.

Inspect summary-first through `emapssn_viewer_data`: get_summary captures an
immutable metadata/membership snapshot. Reuse its snapshot_id with describe_fields,
create_subset, summarize_subset, query_nodes, and read_value. Use describe for each
action's schema. create_subset requires all, visible, or selected scope; an optional
header/metadata/label/selection expression intersects that scope. Empty selection
stays empty. File/residue predicates and command execution are unavailable.
Summarize the population before reading individual records; do not exhaustively
page nodes when aggregates answer the question. query_nodes defaults to 25 rows
and no metadata columns. Request only relevant columns. Node keys are snapshot ID
plus original index; full headers need not be unique. Check mapped/unmapped counts,
missing/invalid values, noise, and overlapping groups before interpreting results.

Data responses default to 16 KiB (max_bytes maximum 64 KiB); use next_cursor to
continue the same snapshot/query. Oversized values have explicit read_value
references: concatenate its JSON-text slices from next_offset, then JSON-decode.
No scientific value is silently shortened. Snapshots expire after 15 idle minutes;
only two and 256 MiB are retained per Viewer, with 32 subsets each. Refresh via
get_summary after expiry or to see edits; never mix different snapshots. Session
listing uses next_offset and does not allocate snapshots. Logs default to 8 KiB.


## Connections, progress, and recovery

Session IDs are full UUIDs; live targeting also accepts exact eight-character
title aliases, case-insensitively. Ambiguous aliases require a full UUID.
`emapssn_viewer_control(action="connect_session")` may omit the ID only when exactly one Viewer is live.
After connecting or launching, omitted IDs use this transport's selection.
Explicit IDs target that call without changing the selection.

`emapssn_viewer_control(action="disconnect_session")` clears selection and leaves the Viewer running.
`emapssn_viewer_control(action="close_session")` terminates it. A closed Viewer makes reads fail;
connect to another session explicitly. Backend shutdown ends the connection. Independent
Viewers remain available for reconnection. A Windows host that forbids independent
launch reports an error; an existing GUI/CLI Viewer can still be connected.

Pipeline and layout jobs share a FIFO queue owned by this server. Use
`emapssn_pipeline(action="list_jobs")` to recover job IDs, and `emapssn_pipeline(action="get_job")` for status,
`failure_message`, and output locations. Backend exit cancels its jobs. Poll at
reasonable intervals; report completion only after `succeeded`, and inspect
relevant outputs before making scientific claims. On failure, read both streams
with `emapssn_pipeline(action="read_log")`, correct the cause, and avoid blindly resubmitting.
`emapssn_pipeline(action="cancel_job")` stops queued or running work; check the returned status.

Both log readers use byte offsets: continue from `next_offset`. `eof` means the
current end of a stream, not job completion. `emapssn_viewer_data(action="read_log")` captures MCP-launched
Viewers in either mode. After disconnect or close, supply a full session ID whose
log location this transport retained. Logs persist on disk, but closed-session
ID lookup does not survive a new transport. Viewers launched without MCP capture
do not offer these logs. Valid file structure, completed generation, and numerical
or scientific correctness are separate conclusions; state the evidence actually
obtained.
