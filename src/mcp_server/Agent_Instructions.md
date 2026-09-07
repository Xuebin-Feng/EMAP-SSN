# EMAP-SSN agent guide

Use this server to prepare sequence data, run supported scientific pipelines,
calculate network layouts, and launch or inspect local Viewers. Choose the entry
point matching the user's task; do not run every discovery tool for every request.

| User's task | Tools and usual order |
| --- | --- |
| Choose a sequence-processing pipeline | `list_pipeline_tools` → `get_pipeline_tool_schema` |
| Check inputs or available computation | `inspect_pipeline_file`, `get_compute_capabilities` |
| Run a pipeline | `validate_pipeline_settings` → `start_pipeline_job` |
| Reuse saved settings | `export_pipeline_settings` or `export_config_settings` |
| Calculate a layout cache | `export_config_settings` (layout) → `start_layout_job` |
| Follow or stop a job | `list_pipeline_jobs`, `get_pipeline_job`, `read_pipeline_log`, `cancel_pipeline_job` |
| Find and inspect an existing Viewer | `list_viewer_sessions` → `connect_viewer_session` → `get_viewer_summary` → `query_viewer_nodes` |
| Launch a new Viewer | `export_config_settings` (viewer) → `get_viewer_settings_schema` → `validate_viewer_settings` → `start_viewer_session` |
| Read Viewer output | `read_viewer_log` |
| Leave or terminate a Viewer | `disconnect_viewer_session` or `close_viewer_session` |

## Run a pipeline

Start with `list_pipeline_tools` when the appropriate pipeline is unknown. Its
entries contain stable `tool_id` values, descriptions, and directory contracts.
A pipeline ID is an argument to MCP tools, not itself an MCP tool name. Pass the
chosen ID to `get_pipeline_tool_schema` for accepted parameters, defaults,
conditions, and an example. Do not invent IDs, script names, or parameter keys.

Use `inspect_pipeline_file` on known input paths when their contents or format
need checking. It inspects a selected file; it does not search directories for
inputs. Use `get_compute_capabilities` when device availability affects a choice,
optionally supplying the pipeline ID. This reports runtime metadata without
benchmarking or executing models; it does not establish that a computation will
fit in memory or succeed.

Prepare exactly one input form: `parameters`, `settings_document`, or
`settings_path`. Optional `directories` accompanies `parameters` only. Use native
JSON numbers and booleans. Call `validate_pipeline_settings`, examine `valid` and
field-specific `errors`, and correct invalid settings before submission. A valid
preview supplies a normalized `settings_document` suitable for
`start_pipeline_job`. Configuration validation does not check every input file,
credential, or hardware requirement.

Submit the chosen pipeline with `start_pipeline_job`, save its `job_id`, and
follow it with `get_pipeline_job`. Submission means queued or running, not
completed. Settings can cause output files to be created or overwritten; align
those choices with the user's request.

## Reuse saved settings

For Viewer launches and layout generation, export first: `export_config_settings`
with `kind` set to `viewer` or `layout` inherits `viewer_settings.json`, including
visual, simulation, and physics preferences. Execute the full exported JSON with
only necessary edits. Never construct minimal JSON from schema defaults instead.
Use `export_pipeline_settings` for saved pipeline settings. Exports create files;
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

First call `export_config_settings(kind="layout")`. Preserve the exported
simulation and physics values (including forces, timestep, convergence, step
limit, and packing) unless changes are required. Pass the edited document/path to
`start_layout_job`; individual arguments can substitute built-in defaults. Layout
JSON differs from pipeline parameters and the flat Viewer document.

Provide the node FASTA and network HDF5, choose physics or UMAP, and use scientific
settings consistent with the network. Ask for missing scientific choices when
they cannot be established from the user's request or inspected metadata. The
start operation validates before enqueueing; there is no separate layout
validation tool. Resolve export errors instead of bypassing saved preferences.
Track the returned job using the shared job tools. After success, inspect the
actual output cache and use its identity and matching inputs to prepare Viewer
settings. Do not assume a layout job opens a Viewer.

## Open and inspect a Viewer

When the user means an existing Viewer, call `list_viewer_sessions` first. Match
the intended session using its identity and cache metadata, then connect. Listing
does not select a session. With multiple candidates, resolve the intended target
rather than arbitrarily choosing one.

For a new Viewer, first call `export_config_settings(kind="viewer")`. To select
another cache, its optional `settings_path` can supply an overlay with
`TARGET_CACHE_PATH` and necessary matching inputs. Keep the full export, consult
`get_viewer_settings_schema`, then call `validate_viewer_settings` and pass its
normalized document to `start_viewer_session`. Validation checks source files and
cache compatibility. Inheritance happens during export; validation must not replace
preserved preferences with a minimal payload. `MSA_FILE=""` disables alignment.

Use the default `normal` mode for a visible Viewer and terminal. Choose `headless`
only when a desktop window is not wanted. Successful launch returns a ready
session and connects this transport to it. Use `get_viewer_summary` for counts,
inputs, cache metadata, and available metadata columns; then `query_viewer_nodes`
for bounded pages of `all`, `visible`, or `selected` nodes. Node offsets and limits
count rows, not bytes. Inspection tools do not execute arbitrary Viewer commands.

## Connections, progress, and recovery

Session IDs are full UUIDs; live targeting also accepts exact eight-character
title aliases, case-insensitively. Ambiguous aliases require a full UUID.
`connect_viewer_session` may omit the ID only when exactly one Viewer is live.
After connecting or launching, omitted IDs use this transport's selection.
Explicit IDs target that call without changing the selection.

`disconnect_viewer_session` clears selection and leaves the Viewer running.
`close_viewer_session` terminates it. Closing the Viewer window clears the
connection automatically; backend shutdown also ends the connection. Independent
Viewers remain available for reconnection. A Windows host that forbids independent
launch reports an error; an existing GUI/CLI Viewer can still be connected.

Pipeline and layout jobs share a FIFO queue owned by this server. Use
`list_pipeline_jobs` to recover job IDs, and `get_pipeline_job` for status,
`failure_message`, and output locations. Backend exit cancels its jobs. Poll at
reasonable intervals; report completion only after `succeeded`, and inspect
relevant outputs before making scientific claims. On failure, read both streams
with `read_pipeline_log`, correct the cause, and avoid blindly resubmitting.
`cancel_pipeline_job` stops queued or running work; check the returned status.

Both log readers use byte offsets: continue from `next_offset`. `eof` means the
current end of a stream, not job completion. `read_viewer_log` captures MCP-launched
Viewers in either mode. After disconnect or close, supply a full session ID whose
log location this transport retained. Logs persist on disk, but closed-session
ID lookup does not survive a new transport. Viewers launched without MCP capture
do not offer these logs. Valid file structure, completed generation, and numerical
or scientific correctness are separate conclusions; state the evidence actually
obtained.
