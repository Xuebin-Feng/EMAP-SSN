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

Plan with `emapssn_pipeline(action="list_tools")`: each `tool_id` lists its inputs and
the directory a relative name resolves in, output names, what happens to an existing
output, and typical next tools; workflow recipes cover common goals (for example,
injecting new sequences instead of recomputing). A pipeline ID is an argument, not an
MCP tool name. Pass it to `emapssn_pipeline(action="get_tool_schema")`
for accepted parameters, defaults, conditions, model availability and an example. Do not
invent IDs, script names, or parameter keys.

Use `emapssn_pipeline(action="inspect_file")` on known input paths when their contents or
format need checking; it does not search directories. Use
`emapssn_pipeline(action="get_compute_capabilities")` when device availability affects a
choice; it does not establish that a computation will fit in memory or succeed.

Prepare exactly one input form: `parameters`, `settings_document`, or `settings_path`.
Optional `directories` accompanies `parameters` only. Use native JSON numbers and
booleans. Call `emapssn_pipeline(action="validate_settings")`, examine `valid` and
field-specific `errors`, and correct invalid settings before submission. A valid preview
supplies a normalized `settings_document` for `emapssn_pipeline(action="start_job")`.
Validation does not check every input file, credential, or hardware requirement.

Submit with `emapssn_pipeline(action="start_job")`, save its `job_id`, and follow it with
`emapssn_pipeline(action="wait_job")`, which returns when the job ends or after at most 50
seconds; call it again instead of polling. Submission means queued or running, not
completed. A finished job's `output_files` lists the files it created, modified or
deleted: take output paths from there. Settings may create or overwrite output
files; align them with the user's request. To time this computer
`emapssn_pipeline(action="start_benchmark")` queues the bundled benchmark; its `result`
lists stage times, devices and report paths.

## Reuse saved settings

For Viewer launches use viewer-control action="export_settings"; for layouts use
pipeline action="export_layout_settings". Both inherit the relevant `viewer_settings.json`
preferences: layout exports include simulation/physics; Viewer exports include
alignment/display preferences. Execute the full exported JSON with only necessary
edits. Never construct minimal JSON from schema defaults instead. Use
`emapssn_pipeline(action="export_tool_settings")` for saved pipeline settings. Exports
create files; explicit output paths must be unused.

When editing JSON, preserve user-defined parameters unless they invalidate the job. For
a different input file, replace only that field plus necessary dependent fields (paired
embedding and network files may need matching sources). Do not reset unrelated settings
or guess dependent files. Explain any required changes.

When reporting calculations or generations, always list the effective parameters and
input files actually used, including preserved values and applied defaults, from the
executed settings rather than just the requested edits.

Exporting executes nothing, and an exported layout cache filename is a preview,
not a reservation. Viewer exports may select the newest compatible cache: verify it is
the intended one. Execution documents are not silently refreshed from personal settings.
Relative input paths follow each tool's directory rules, not the settings file's folder.

## Calculate a layout

Choose the edge filter from evidence: `emapssn_pipeline(action="network_statistics")`
scores a network as the layout will (pass the node FASTA, ALIGNMENT_SCORE and NORM_MODE)
and reports kept edges, clusters and isolated nodes per cutoff. Prefer the user's
criterion when given. `emapssn_pipeline(action="get_layout_schema")` explains every field.

Then call `emapssn_pipeline(action="export_layout_settings")` with node_fasta_file,
input_hdf5 and the filter in parameters. Keep the exported simulation and physics values
unless changes are required, and pass the document to
`emapssn_pipeline(action="start_layout_job")`; individual arguments substitute built-in
defaults. Layout JSON differs from pipeline and Viewer JSON. Resolve export errors instead of bypassing saved preferences.

layout.LAYOUT_DIMENSIONS is 2 (desktop Viewer) or 3; layout.LAYOUT_SEED is a
non-negative integer, default 42, or null (unseeded). Ask for missing scientific choices
when they cannot be established from the request or inspected metadata. The start
operation validates before enqueueing; there is no separate layout validator. The
finished job's `result` reports nodes, edges, effective threshold, clusters and isolated
nodes. Inspect the cache and use its identity and matching inputs to prepare Viewer
settings. Do not assume a layout job opens a Viewer.

## Open and inspect a Viewer

When the user means an existing Viewer, call `emapssn_viewer_data(action="list_sessions")` first. Match
the intended session using its identity and cache metadata, then connect. Listing
does not select a session. With multiple candidates, resolve the intended one;
never choose arbitrarily.

For a new Viewer, first call `emapssn_viewer_control(action="export_settings")`. To select
another cache, its optional `settings_path` can supply an overlay with
inputs.TARGET_CACHE_PATH and necessary matching inputs. Overlays require
schema_version=2 and kind="viewer", with edits inside the named sections. Keep the full export, consult
`emapssn_viewer_control(action="get_settings_schema")`, then call `emapssn_viewer_control(action="validate_settings")` and pass its
normalized document to `emapssn_viewer_control(action="start_session")`. Validation checks source files and
cache compatibility. Inheritance happens during export; validation must not replace
preserved preferences with a minimal payload. alignment.MSA_FILE="" disables alignment.
When the selected MSA does not contain alignment.ALIGNMENT_REFERENCE, validation
preserves the requested reference and offset. The Viewer loads the MSA, warns in
its startup log, and uses occupancy-based numbering with alignment offsets inactive.
Successful validation does not guarantee reference-based numbering; inspect the
Viewer summary's requested/resolved reference and startup log after launch.
A nonempty reference with an empty MSA_FILE remains invalid.

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
only when a desktop window is not wanted. A ready launch connects this transport;
status="starting" means still loading: call `emapssn_viewer_control(action="wait_session")`
with its launch_id, never relaunch.

Inspect summary-first through `emapssn_viewer_data`: get_summary captures an
immutable metadata/membership snapshot. Reuse its snapshot_id with describe_fields,
create_subset, summarize_subset, query_nodes, and read_value. create_subset requires all, visible, or selected scope; an optional
header/metadata/label/selection expression intersects that scope. Empty selection
stays empty. Residue predicates need an include_alignment=true snapshot; file
predicates are unavailable. Execute user commands through the separate command portal when requested.
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

Pipeline, layout and benchmark jobs share a FIFO queue owned by this server. Use
`emapssn_pipeline(action="list_jobs")` to recover job IDs, and `emapssn_pipeline(action="get_job")` for status,
`failure_message`, and output locations. Backend exit cancels its jobs. Wait with
wait_job; report completion only after `succeeded`, and inspect
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


## Viewer command portal

Discover summaries, arguments, choices/aliases and syntax with get_command_catalog. For detailed help, use
`emapssn_viewer_data(action="get_command_catalog", arguments={"command":"reset"})`.
This is read-only and does not require execute_commands or a queued help command.
General catalog entries have help=null; syntax lists extracted usage signatures
and may be empty when source help has no extractable signature.
Submit emapssn_viewer_control(action="execute_commands") with submission_id; reuse it on retries.
Follow its next_step (tool, action, arguments) to poll the originating session.
get_command_request and read_command_output accept exactly one nonempty request_id
or submission_id. IDs are Viewer-local; after reconnecting, supply session_id
(list_command_requests recovers IDs).
Unknown/evicted submissions return errors without executing work.
Poll get_command_request and inspect per-command messages for outcome summaries.
Execution completion is indicated by status (succeeded, failed, or cancelled);
the complete flag describes pagination only and does not indicate execution completion.
Use read_command_output for additional diagnostics or full printed content,
not as a mandatory step after every success. Submission messages for background
work do not prove job completion; continue checking status and jobs.
Batches stop on failure. Awaiting_user_input requires interaction.
Use capture_view for current PNG/HUD verification. For visual node fields, get_summary(include_visual=true), then query_nodes(visual_fields).

## Structured residue analysis

Capture with `get_summary(arguments={"include_alignment":true})` and retain its
snapshot_id. Alignment data stays in Viewer memory, subject to the snapshot budget.
Call `emapssn_viewer_data(action="get_residue_distribution")` with snapshot_id, positions=["1883"] and optional
subset_id or group_by="cluster"/"group". Counts use mapped network nodes; gaps
remain in the denominator and unmapped nodes are excluded. Fractions are 0..1;
zero denominators produce null fractions. Group membership can overlap.
Positions are explicit displayed labels, including negatives and insertions, not ranges.
On alignment snapshots, create_subset accepts residue predicates with the existing
Boolean grammar. No live selection changes are needed.
For compact node pages, query_nodes(fields=["node_id"]) returns only node IDs.
Omitting fields preserves existing rows; columns/visual_fields select container contents.
read_command_output captures MCP-submitted commands even in attached Viewers.
read_log requires launch-time process-log capture and is not a substitute.
