# Changelog

All notable changes to EMAP-SSN are documented in this file. The project uses
[Semantic Versioning](https://semver.org/); interfaces and persisted formats may
still change before version 1.0.0.

## [Unreleased]

### Added

- MCP `emapssn_viewer_control(action="wait_session")` keeps waiting for a Viewer launch
  and connects when it is ready. `start_session` takes an optional `ready_timeout`
  (default 45 s, at most 600), and `close_session` accepts the launch's `launch_id`,
  which also stops a Viewer that is still loading. MCP server version 0.12.0.
- `emapssn_pipeline(action="inspect_file")` reports a layout cache's saved state —
  cluster count and noise nodes, groups, the last clustering parameters, and metadata
  columns — and its layout dimensions. Viewer `export_settings` reports the same state
  for the selected cache and for the folder's other caches, so a saved figure cache
  can be chosen deliberately.
- `get_command_catalog` works without a running Viewer; it then reads the installed
  command sources.
- The ESM3 structure worker (`esmfold_worker.py`) gains `--skip-existing`, which keeps
  structures already on disk so an interrupted run can resume.
- `Parse_BLAST_Output.py` records DIAMOND provenance. DIAMOND output written with
  `--header verbose` declares its version and command line; the importer stores them,
  like a BLAST+ program line, in the network's `search_program`, `search_version` and
  `search_invocation` attributes and names the network `<name>_[DIAMOND]_EValue.h5`.
  `model_name` stays `BLAST`, so the Viewer still loads it as an E-value network.
- The BLAST/DIAMOND importer warns when an all-vs-all search was truncated: when a
  recorded DIAMOND command's `--max-target-seqs` (25 by default) was reached or
  `--top` was used, or, for headerless files, when queries share a target count while
  being reported as hits by more queries than that count. It also warns when FASTA
  records never appear as a query although others report self hits. Warnings are
  stored in `import_warnings`, and the diagnostics report queries observed and the
  largest target count.
- `inspect_file` detects DIAMOND `--header verbose` files as BLAST tables, reports
  their program, version and command, and warns when the command lacks `-k 0`. For an
  imported network, it reports the recorded search program and import warnings.
- **Automatic layout step size.** An Auto button to the left of the Step Size field
  (setting `AUTO_DT`, off by default) lets each simulation stage use the largest step
  that keeps it stable, and greys out the field. The step is half the stability limit
  for the stage's springs, bounded through their largest `d_u + d_v`, plus one
  repulsive contact at its stiffest (`COULOMB_K` and `MAX_FORCE_LIMIT`). Without the
  contact term, pairs and triangles sized for their springs alone kept passing
  through each other's repulsive cores. `MAX_STEPS`, `RMSD_THRESHOLD` and
  `RMSD_WINDOW` still count steps. On the 4,033-sequence test network the stages ran
  at 0.0097 to 0.0148 and the batch of small components at 0.083, with the same run
  time and edge lengths as `DT` 0.01. Layout documents without `AUTO_DT` load with it
  off. The VR Config has the same button.

### Changed

- **A slow Viewer launch is handed back instead of stopped.** When `start_session`'s
  wait ends while the Viewer is still loading, it returns `status: "starting"` with the
  `launch_id`, the Viewer's last output line, its process-tree CPU time and the log
  paths; the Viewer keeps loading. Cancelling the call, including a client-side tool
  timeout, also leaves it running. The Viewer now logs when it validates its inputs
  and when it builds the network display.
- Faster Viewer startup on large networks: edges are drawn without a per-edge Python
  loop, and the top-percent cutoff uses a partial sort. On a 12,927-node network with
  4.18 million displayed edges, headless readiness fell from 14.0 s to 9.6 s on the
  test machine, and edge drawing from 5.9 s to 0.15 s. Layout generation shares the
  faster network preparation; cutoffs are unchanged.
- Less memory while the Viewer, the VR viewer and layout generation read a network.
  Only the columns the settings use are read: the alignment-length column only for
  `alignment_length` normalization. Scores are also normalized without a second
  full-size copy. On the same network (83.5 million pairs), peak memory during loading
  fell from 1,519 to 1,201 MiB with global scores and alignment-length normalization,
  and from 1,998 to 1,520 MiB with local scores normalized by the shorter sequence.
  Scores, cutoffs and edges are unchanged. The Config window's network statistics and
  score histogram also skip the alignment-length column when they don't divide by it,
  and keep pair indices in the file's 16- or 32-bit type instead of widening them to
  64 bits; on the same network the statistics' peak fell from 3,520 to 2,564 MiB, with
  an identical report.
- The ESM3 structure worker no longer notifies a Viewer unless `--action-url` is
  given (it used to post to port 8000), and deletes its input JSON only with the new
  `--delete-input`, which the Viewer's `esmfold` command passes. PDB files are written
  atomically, with a warning when a structure is replaced or two headers map to the
  same file name.
- The BLAST importer's "not present in the FASTA manifest" error says when the
  header is the first word of a FASTA header that contains spaces, which is all
  BLAST+ `qseqid`/`sseqid` and DIAMOND report, and how to keep complete titles. A
  `# Fields:` comment must declare an E-value at the selected E-value column, and a
  DIAMOND `--header simple` column-name row is rejected with advice.
- **Output names stay in their folders.** `select save`, `meta download` and `print`
  take a plain file name and write only into their configured folders, as `save`,
  `logo` and `label` already did. A name with a directory, `..`, a drive (`C:`), a
  network share (`\\host\share`) or a `:` stream suffix is refused. The web agent and
  MCP clients issue these commands too, and on Windows a network-share name makes the
  system offer the user's credentials to that host. `meta download` writes `.csv` and
  `.xlsx` only; other extensions are refused before anything is written.
- **CPU layout generation uses several cores.** The SSN physics kernel computes
  repulsion in parallel on all but two logical CPUs, as the `logo` command already
  did (set `NUMBA_NUM_THREADS` to choose the count). It also visits only pairs within
  a component and no longer allocates temporaries for every node on every step. On
  the test machine (20 logical CPUs, so 18 threads), a step took 15 ms instead of
  105 ms on a 10,000-node component, and 0.074 ms instead of 1.8 ms on a batch of 100
  small components. Coordinates are bit-identical to the serial kernel on any thread
  count. UMAP layouts with a `LAYOUT_SEED` still run on one core, because umap-learn
  makes seeded runs serial to keep them reproducible.
- Neighbor-joining guide trees in `Embedding_MSA.py` and the edge filter of the
  `jaccard` mode of `cluster` and `subcluster` also run in parallel on all but two
  logical CPUs; neighbor-joining bootstrap workers split those threads between them.
  A 3,000-sequence neighbor-joining tree took 0.74 s instead of 5.9 s, and filtering
  3 million edges 0.26 s instead of 3.7 s, with identical trees and filters.
- **Embedding MSA merges large clusters in seconds, not minutes.** Each merge rebuilt
  every aligned sequence one character at a time, at a cost that grew with the
  number of sequences times the square of the alignment width. On a 44,127-sequence
  network the merge stage took 6 h 23 min of a 7 h 47 min run, and each of its last
  merges took 1.5 to 3.3 minutes. Alignments are now byte matrices that a merge
  fills a whole column at a time. On the test machine, merging 44,126 sequences with
  one more into 9,684 columns took 0.15 s instead of 212 s. Leaf embeddings are read
  through the open embeddings file instead of reopening it for every sequence.
  Aligned FASTA files and merged profiles are unchanged bit for bit. A sequence that
  contains non-ASCII characters is now rejected before the guide tree is built.
- **Packed 2D layouts are tighter.** A component's clearance box that ends exactly on
  a packing-grid cell border no longer claims the next cell, and `PACKING_GRID_SIZE`
  defaults to 10 instead of 20, also in the VR Config, where it sets the gap between
  shells. The grid size now only sets how finely components are fitted together:
  drawings of different components always stay at least half of `PACKING_PADDING`
  apart. A 12,927-node BLAST network that packed 440 units across at grid size 20
  packs 240 across at the new default and 165 at 2.5. Packed positions differ from
  earlier versions; saved caches keep theirs.
- **Tiled embedding alignment is about 1.8 times faster.** `Align_Similarity_Matrix`
  and `Network_Injection` built every padded GPU batch with one copy per target and
  sent the target lengths from unpinned memory, which made the CPU wait for the GPU on
  every batch and left the GPU half idle. Each tile's embeddings are now uploaded once
  into a packed GPU buffer that every batch gathers from in one step, lengths travel
  through pinned memory, one CPU task aligns a whole batch instead of one task per
  pair, and tiles no longer wait for the GPU queue to empty when they change. On the
  test machine (RTX 4070, i5-13600K), an MCP job on 12,927 sequences took 31.4 s per
  500,000-pair batch instead of 56.9 s. The old code's speed also varied with CPU load
  (up to 88 s per batch in another session); the new code is limited by the GPU.
  Scores and alignment lengths are unchanged bit for bit for a given execution plan.
- **Faster alignment scoring when the CPU sets the pace.** A float32
  dynamic-programming kernel aligns pairs about 2.2 times faster per core when float32
  is provably exact: a global gap penalty of 0 with a local gap penalty of 0 or a
  negative power of two, which includes the defaults (0 and -2). Other gap penalties
  use the previous kernel. With 2 CPU workers, the first batch ran at 7,255 instead of
  3,326 pairs per second, with identical scores. This matters with GPUs much faster
  than the test machine's, where the CPU alignments limit throughput.

### Fixed

- MCP `start_session` terminated a healthy Viewer after a fixed 30 s. On a
  12,927-node network loaded from OneDrive, the Viewer was stopped while building its
  display, seconds from publishing its session. A readiness probe that timed out while
  the Viewer's Qt thread was busy also stopped the launch; the probe is now retried.
- MCP pipeline jobs inherited the server's standard input, which carries the MCP
  protocol; a job that prompted for input could consume protocol messages.
- On Windows, cancelling an MCP pipeline job, or closing the MCP server while a job
  ran, could leave processes the job had started running, such as the worker processes
  of an alignment with several `WORKERS`. The server ended only the process it had
  launched, which in a virtual environment is the `python.exe` redirector. The tool's
  interpreter stopped with it, but the interpreter's own child processes did not. The
  job's whole process tree is now ended; on Linux and macOS, the job's process group
  already was.
- Embedding MSA and embedding search (SSEARCH), run through MCP or a settings file,
  accepted local scores with `alignment_length` normalization, dividing each local
  score by its own alignment length; the Tools window, the Viewer and layout generation
  already refuse that pairing. Both tools now stop with a configuration error before
  opening their inputs, and their MCP schemas require `shorter_sequence`,
  `longer_sequence` or `average_sequence` with local scores.
- Embedding MSA, embedding search (SSEARCH) and pairwise embedding alignment (PWA), run
  with a hand-written settings file, accepted an `ALIGNMENT_SCORE` (MSA) or
  `ALIGNMENT_MODE` (SSEARCH, PWA) other than `global` or `local`. SSEARCH ran a global
  search, with the global gap penalty, for any value but `local`, such as `Local`.
  Embedding MSA built its guide tree from local scores for any value but `global`, such
  as `Global`, even with `alignment_length` normalization. PWA ran a local alignment,
  with the local gap penalty, for any text but `global`, such as `Global`, while its
  report printed that text as the mode. All three tools now stop with a configuration
  error that names the setting, before opening their inputs.
- Embedding MSA kept its noise-perturbed guide-tree cache (a memory-mapped distance
  matrix of about 4 bytes per sequence pair, 3.9 GB for 44,000 sequences) in the
  project's default `Input_Files/Multiple_Alignments` folder whenever it ran through
  MCP, from the command line or from the Tools window, whatever `MSA_DIR` was set to.
  Such a run recreated that folder if it had been removed and could fill the project's
  drive. The cache now goes to the run's `MSA_DIR`, or to `SAFE_TEMP_DIR` when a
  settings file sets one.
- On Linux and macOS, a normal-mode MCP `start_session` ran its terminal program on the
  server's own standard streams: it inherited the MCP protocol's input and output, and
  its errors (an emulator's "cannot open display", or macOS refusing control of
  Terminal) never reached the launch logs. A terminal that could not start the Viewer
  therefore surfaced only as a timeout. The terminal program now gets no input and
  writes to the launch's `stderr.log`. A launch fails, with that output, as soon as its
  terminal exits with an error, or 60 s after it started when its terminal exited
  cleanly without starting the Viewer.
- On Windows, a Viewer started by MCP `start_session` failed any command that printed
  text outside the Windows ANSI code page (cp1252 on most Western systems), such as a
  selection error echoing an expression with `α-amylase` in it, or the "LLM Agent
  Activated: … → …" line: the command stopped with "'charmap' codec can't encode
  character". The Viewer printed to a pipe in normal mode and to its log files in
  headless mode, which Python writes in that code page, while `read_log` and the
  Viewer's console window read UTF-8, so even `é` came back as `�`. The Viewer now
  writes UTF-8 to pipes and files, whatever started it, and escapes characters a
  terminal's encoding lacks instead of failing. The normal-mode console window also
  lost or garbled a character that a pipe read split in two; it now shows it whole.
- On Windows, the `run` command mangled non-ASCII text in the commands that a Python
  command script printed, however the Viewer was started. Python wrote the script's
  output to the Viewer's pipe in the Windows ANSI code page (cp1252 on most Western
  systems), while the Viewer read it as UTF-8. A script printing `α-amylase` failed
  with "'charmap' codec can't encode character", and `é` vanished from a typed `run`
  (`select "café"` ran as `select "caf"`, which also selects "caffeine") or became
  `�` in a request from MCP or the agent page. Scripts now print UTF-8 whatever the
  Viewer's environment says, and a byte that is still not UTF-8 shows as `�` instead
  of disappearing.
- The `run` command misread `.txt` command files saved in common Windows encodings.
  With a UTF-8 byte-order mark (a Notepad option, and PowerShell 5.1's
  `Set-Content -Encoding UTF8`), the first command failed as unknown. A UTF-16 file
  (PowerShell 5.1's `>` and `Out-File` default) became NUL-laced garbage commands. An
  ANSI file (older Notepad's and PowerShell 5.1 `Set-Content`'s default) lost its
  non-ASCII characters, so `select "café"` ran as `select "caf"`, which also selects
  "caffeine". `run` now reads a `.txt` file as Notepad does: a byte-order mark (UTF-8,
  UTF-16 or UTF-32) names the encoding, and a file without one is UTF-8 when it
  decodes as UTF-8 and is otherwise read in the system's ANSI code page (cp1252 on
  most Western systems), with a note in the terminal. A byte that still doesn't
  decode shows as `�` instead of disappearing. The VR viewer's `run` reads files the
  same way.
- On Windows, Config and Tools opened from the EMAP-SSN and EMAP-SSN Tools shortcuts,
  and MCP layout jobs, failed when they printed text outside the Windows ANSI code page
  (cp1252 on most Western systems). Their output goes to a log file, which Python
  writes in that code page. Tools could not run any tool from a project folder such as
  `Projekt-α`: Run reported "'charmap' codec can't encode character" and started
  nothing. Config closed when it found two compatible cache folders for the selected
  inputs and their paths had such a character, and a `start_layout_job` job reading or
  writing such a path failed. The terminal that shows the log after a failure read it
  as UTF-8, so even `é` came back as `�`. Config and Tools now write UTF-8 to files and
  pipes, whatever started them, as the Viewer does, and escape characters a terminal's
  encoding lacks instead of failing.
- `inspect_file` reported valid 3D layout caches (for the VR viewer) as invalid.
- `capture_view` on a headless Viewer failed with a bare OpenGL error; it now explains
  that headless Viewers cannot render on Windows and that normal mode can.
- On Windows, an ESM3 structure run started through MCP or the web agent could report
  a saved structure as failed (`[WinError 5] Access is denied`) when it updated its
  status file while the Viewer, or another program, was reading it. When this hit the
  final status, the Viewer reported that the worker exited before reporting a result.
  Status updates now wait for the reader, and one that still fails prints a warning
  without failing the structure.
- `meta download` and the metadata spreadsheet's Excel export failed for a file name
  ending in upper-case `.XLSX` ("No engine for filetype: 'XLSX'").
- The agent panel inserted HTML from model replies, including chat history reloaded
  from a layout's `agent_history.json`, as live markup, so a reply could run script in
  the Viewer's web pages, which can send Viewer actions and read the saved model cards
  with their API keys. Raw HTML in a reply is now shown as text, and links and images
  keep only http(s) and relative URLs. The bundled pages are also served with a
  Content-Security-Policy that runs only their own scripts; images from other hosts in
  a reply are no longer loaded.
- A hand-edited layout cache could make `export` write outside its folder, in both the
  desktop and VR viewers. The viewers restore a cache's group labels and last clustering
  parameters unchecked, and `export` names files after the labels and the cluster folder
  after the parameters, so `..\`, an absolute path or a `\\host\share` network path in
  either wrote FASTA files and created folders elsewhere. `export` now refuses such a
  label or such parameters, names what it refused, and writes nothing. Names the `group`
  and `cluster` commands produce are unaffected; a group named `..` still exports as
  `...fasta`.
- `Align_Similarity_Matrix.py` printed "✅ Compilation complete!" and exited with
  code 0 when a batch file could not be read or merged into the final network, so MCP
  reported the job as succeeded. The network then lacked that batch's edges or held rows
  with zero scores: the batch's own pairs, or `(0, 0)` pairs. Because the file existed,
  running the job again only reported "Job already done". A batch that cannot be read or
  merged now stops the job with exit code 1 and names the file, and the network is
  written to a `.partial` file that replaces the final one only after every batch has
  merged. Successful runs write identical networks.
- `Align_Similarity_Matrix.py` also exited with code 0 when it stopped before
  aligning, so MCP reported jobs that wrote nothing as succeeded. That happened when
  no embeddings file was selected or it was missing or invalid, when
  `ACCELERATOR_PRECISION` or `EXECUTION_MODE` could not be used (an unknown value,
  failed BF16 validation, TF32 without a CUDA device, or tiled mode without a
  compatible accelerator), and when an existing network could not be read or was
  computed at a precision the settings exclude. These now exit with code 1; "Job
  already done" still exits with code 0.
- Embedding MSA ran out of memory on large complete networks. Filtering stored every
  edge as a Python tuple (about 130 bytes each), so a 44,127-sequence network with
  973.6 million edges needed about 150 GB and stopped with `MemoryError` on a 96 GB
  PC. Edges are now filtered as whole arrays, and arrays are freed once used. Bootstrap
  workers pass SciPy a float64 matrix directly instead of a float32 one it must
  convert, and tree distances are summed in place. That run should now peak at about
  51 GB with `WORKERS` 3, about 16 GB per worker; guide trees are unchanged bit for
  bit.
- `Sparse_MSA_Converter.py` exited with code 0 when it converted nothing: when its
  input FASTA was missing, validation rejected the alignment, or the HDF5 save failed.
  An MCP job therefore reported `succeeded` (seen when step 4 failed and the queued
  converter found no alignment). It now exits with code 1. With `CONVERT_ALL`, it
  exits with code 1 when any alignment fails, after converting the rest, or when
  `MSA_DIR` holds no `.fasta` files.
- `Network_Injection.py` stopped every run with `TypeError` before aligning anything
  (v0.2.0 and v0.3.0): it checked the precision of an existing output network before
  setting that network's path. Past that check, CPU runs failed too, because the
  worker processes never received the gap penalties read from the input network.
  Both are fixed, so runs complete.
- `Network_Injection.py`'s final compile step skipped a batch file it could not read,
  printing only a warning, and skipped one lacking `i` or `j` without one. That batch's
  pairs stayed in the network with zero scores and lengths, which look like real edges,
  and the job exited with code 0. Such a batch now stops the job with exit code 1 and
  names the file. The network is written to a `.partial` file that replaces the final
  one only once complete, so a failed write no longer leaves a truncated network or
  replaces an earlier one. Networks are byte-identical to the old code's output with
  both failures above bypassed.
- `Network_Injection.py` also exited with code 0 when it stopped with "Cannot start
  Network Injection", so MCP reported jobs that wrote nothing as succeeded. That
  happened when no existing network or new embeddings file was selected, when
  `EXECUTION_MODE` was unknown or forced tiled mode without a compatible accelerator,
  and when the new embeddings file was incomplete or could not be read. These now exit
  with code 1 and print the same message.
- Layout generation ran out of memory while packing components. To find the grid
  cells a component needs, packing stored a point every quarter cell along every edge,
  about 128 bytes each, so memory grew with the drawn length of all edges. A
  44,127-sequence layout whose largest component (40.5 million edges) the simulation
  had left scattered across its box reached 82 GB before it was stopped. A
  component's cells are now marked directly from its nodes and edges, and the
  component search and placement run in compiled code. On the test machine, a
  network with a scattered 12,289-node component (4.2 million edges averaging 590
  units) packed in 1.7 s using about 140 MB. Edges are also grouped by component as
  arrays rather than one Python tuple each: generating a 12,927-node layout with 4.18
  million edges peaked at 1.3 GiB instead of 2.6 GiB, with simulated coordinates
  unchanged bit for bit.
- `PACKING_GEOMETRY` `Circle` let packed components overlap. Footprints worked out on
  square cells were placed on a hexagonal lattice whose rows are 13% closer together,
  so on a 12,927-node network nodes of different components came as close as 0.09
  units, against a clearance of 5. Circle now uses the same square cells as Square,
  filled in order of distance from the centre, so the packed layout still grows as a
  disc.
- Layout generation could blow up a dense component instead of relaxing it, and
  logged the stage as finished. The force simulation is stable only while
  `SPRING_K` × λ × `DT`² < 4 − 2 × `DAMPING` × `DT`, where λ, the largest eigenvalue of
  the graph Laplacian of the stage's springs, is close to the busiest node's spring
  count. With `DT` 0.01, the largest component of the 44,127-sequence layout passed
  that limit in progressive stages 4 and 5 (λ 8,802 and 10,929). Its nodes ended up
  bouncing between the walls of the layout box, and the log reported "Plateau
  Reached". Each stage now lowers `DT` when needed, to 0.85 of the limit for the
  largest `d_u + d_v` over its springs (an upper bound on λ), and scales `MAX_STEPS`,
  `RMSD_WINDOW` and `RMSD_THRESHOLD` so that the stage keeps its simulated time and
  convergence speed. The log says when it does. Stages that were already safe keep
  `DT`, and their layouts are unchanged bit for bit; at `DT` 0.01 that covers every
  stage of the 4,033- and 12,927-sequence test networks. At `DT` 0.05, which crosses
  the limit in every stage of the 4,033-sequence network, its edges had a median
  length of 397 units before and 1.5 after (1.3 at `DT` 0.01).
- The layout log called a stage whose RMSD was growing "Plateau Reached", and said
  nothing when a stage used up `MAX_STEPS`. An RMSD that rose by more than 1% over
  the last window is now reported as "Not settling", and running out of steps as
  "Step limit reached". The log also warns when active nodes end a stage on the
  layout boundary, where a diverged simulation leaves them. Stages stop at the same
  step as before.
- In a narrow Config or VR Config window, where the Alignment Reference ID, Min
  Occupancy % and Alignment Offset fields stack, the Alignment Offset field stayed
  100 px wide instead of reaching the right edge like the others. It now does; in a
  wide window it keeps its 100 px beside the other two.
- `Network_Extraction.py` left whitelist sequences out of the sub-network. It read the
  whitelist FASTA as plain UTF-8, so a file saved with a byte-order mark, as Windows
  Notepad can write, lost its first record without any message. It also compared
  headers as written, while networks store them sanitized, so a whitelist taken from
  the FASTA the network was built from missed every record whose header sanitizing
  changes, such as one with a description after a space, brackets or a `/`. The tool
  now reads the whitelist with the shared FASTA reader, sanitizes each header with the
  same rules as the network's headers, and reports how many whitelist headers matched
  no network header.
- `@file@` header lists in Viewer selection expressions left nodes out without any
  message. A list saved with a byte-order mark, as Windows Notepad can write, lost its
  first entry. Entries were compared as written, while networks store headers
  sanitized, so a header copied from the FASTA the network was built from matched
  nothing when sanitizing changes it, such as one with a description after a space.
  `[NCBI]` and `[PDB]` lists missed every node whose accession or PDB ID is followed by
  a description, because sanitizing turns the space after the ID into `_`, which the
  ID search read as part of the ID. A list that was not UTF-8 lost the characters it
  could not decode, and a UTF-16 list matched nothing. Lists are now read as UTF-8
  with or without a byte-order mark, each entry is sanitized like a network header,
  and an ID ends at `_` as it does at a space. A list in another encoding is refused
  with a message, and the terminal lists the entries that matched no node.
- The Config and VR Config accepted a saved profile named "(custom)", "(default)" or
  "(new)". The profile selector listed it under the same text as the built-in entry
  and could not tell the two apart. These names are now reserved like "custom",
  "default" and "new", and an existing profile file with one of them no longer
  appears in the selector; rename the file to use it again.
- `label` reported each failure to MCP clients twice, once with and once without the
  `Error:` prefix, for example when an argument was invalid or no clusters or groups
  existed. Each failure is now reported once, as the console shows it.
- The Config and VR Config could not save while a BLAST network was selected: **Save**
  reported "invalid value for ALIGNMENT_SCORE" and **Save & Run** stopped before
  launching. A BLAST network blanks the Alignment Score Mode and Normalization Mode
  fields, which BLAST scores do not use, and the blank fields were saved as empty text.
  Saving now keeps the choice the fields showed before they went blank, or that of a
  profile loaded since. Selecting an alignment network again restores that choice
  instead of resetting it to global and alignment_length.
- On Linux, GPU detection gave a GPU the kernel driver of a device `lspci` lists
  after it, usually the GPU's own HDMI audio function, so the detection report
  showed `snd_hda_intel` as the driver of an NVIDIA or AMD card. An Intel Arc GPU
  with no driver bound could be reported eligible for the XPU backend rather than
  provisional, because the next device's driver counted as its own. Device names
  from `lspci` also kept their trailing PCI ID and revision, such as
  `[10de:2684] (rev a1)`; they now end with the model name. Because the name is part
  of the saved hardware profile, the first launch after updating on Linux with an AMD
  or Intel GPU, or an NVIDIA GPU without a working `nvidia-smi`, re-validates the
  installed PyTorch backend once without reinstalling it.

### Removed

- The Viewer's `GET /api/mcp/v1/nodes` inspection route, which returned a page of
  live node data. No client used it: the MCP server reads node pages from captured
  snapshots through `query_nodes` on `POST /api/mcp/v1/data`.

## [0.3.0] - 2026-10-01

### Highlights

- **3D layouts and reproducible seeds.** The physics and UMAP engines can generate 2D
  or 3D coordinates (`LAYOUT_DIMENSIONS`), and `LAYOUT_SEED` (default `42`) seeds both
  engines, so UMAP layouts and CPU physics layouts reproduce exactly. 3D caches serve
  the new optional VR viewer in the `opt_vr` submodule; the desktop Viewer keeps using
  2D layouts.
- **Protein properties as metadata.** New layout caches carry `Length`, `kDa`, `pI`,
  and `GRAVY` columns computed from the input FASTA.
- **Residue-level inspection over MCP.** Viewer snapshots can freeze the displayed
  alignment, enabling paged residue distributions and residue predicates in subsets.
- **No bundled Python wheels.** ESM and Transformers are now installed from PyPI at
  their published versions. Earlier releases redistributed a self-built ESM 3.3.0
  wheel, because the ESM releases on PyPI at the time carried the Cambrian license,
  and the reproducible Transformers fork labeled `4.57.6+biohub.3a8956f`, which ESM
  3.3.0 declared as a direct Git dependency. The MIT-licensed ESM 3.4 is published on
  PyPI and no longer depends on the fork, so the fork is obsolete. This removes the
  project's Python wheel-redistribution obligations entirely.
- **Python 3.13.** The managed environment target moves from 3.12 to 3.13, which
  ESM 3.3.0's `Requires-Python <3.13` had ruled out.
- **One ROCm backend instead of four.** Windows and Linux now share a single ROCm 7.14
  profile from AMD's multi-architecture channel.

### Upgrade Notes

- **`color` scales use a trailing `x`.** Write `2x`, `0.5x`, or `0x`. The prefix form is
  removed: `x2` now selects residue X at position 2, so saved commands that scale with
  `x<number>` must be rewritten. In `color` and `spectrum`, uppercase `C<number>`
  always selects the cysteine at that position and is never read as a Matplotlib
  `CN` color.
- **Check saved alignment references.** `ALIGNMENT_REFERENCE` and `reference` now
  resolve the target against the network headers — an exact header first, then a
  leading identifier, then a substring or wildcard match — and the alignment anchors
  only on that header (see Fixed). v0.2.0 anchored on the first MSA row that matched
  by exact name or by substring in either direction. A saved reference can therefore
  anchor on a different sequence, or stay inactive when the MSA lacks it, which leaves
  the alignment in occupancy numbering. Positions in `query`, `logo`, `label`, and
  residue expressions change with it, and the `logo` axis now names the header the
  alignment anchored on instead of the text that was typed.
- **ESM-2 3B and 15B are no longer supported.** `esm2_t36_3b` and `esm2_t48_15b` were
  removed, so `esm2_t33_650m` is now the largest ESM-2 model. Saved Tools settings
  that name either model must select another one. The Tools window shows the saved
  name as "Unavailable saved model [name]" and will not run until another model is
  selected.
  Headless and MCP runs reject the name. Embedding files created with either model
  can no longer be resumed, extended by `Embedding_Injection.py`, or queried with new
  sequences in `Embedding_SSEARCH.py` or `Embedding_PWA.py`, because no plugin can
  load the model.
- **ESMC and ProstT5 embeddings change slightly.** ESMC 300M and 600M now run in
  float32 on every device. In v0.2.0 their precision depended on how the model was
  loaded: an explicitly selected GPU, and the injection, SSEARCH, and PWA tools on a
  GPU machine, loaded bfloat16 weights; automatic device selection kept float32
  weights but computed under bfloat16 autocast on CUDA and ROCm GPUs; CPU runs, and
  XPU or MPS runs under automatic selection, used float32. GPU-computed ESMC
  embeddings therefore generally do not match v0.2.0 exactly. ProstT5 now loads the
  unrounded float32 `Rostlab/ProstT5` checkpoint, an 11.3 GB download, instead of the
  5.6 GB `Rostlab/ProstT5_fp16`, so its embeddings differ on every device. Results
  derived from these embeddings change accordingly. The weights now come from the
  Hugging Face repositories `biohub/ESMC-300M`, `biohub/ESMC-600M`, and
  `Rostlab/ProstT5` and are downloaded again on first use. ESM-2, ProtBERT, and Ankh
  keep the checkpoints and float32 compute precision they had in v0.2.0.
- **Regenerate, don't resume, v0.2.0 ESMC and ProstT5 embedding files.** Embedding
  files record only the model name and saving mode, so resuming
  `Generate_Embeddings.py` on, or injecting sequences into, an `esmc_300m`,
  `esmc_600m`, or `prost_t5` file from v0.2.0 would silently mix embeddings computed
  at a different precision (ESMC) or from a different checkpoint (ProstT5).
- **Regenerate layout caches to get the new metadata columns.** Caches written by
  v0.2.0 still open, but the Viewer no longer synthesizes a `Length` column for a
  cache without stored metadata, so such caches open without metadata columns until
  they are regenerated.
- **Windows ROCm requires Windows 11 25H2.** On earlier Windows 11 builds, v0.2.0
  served `gfx1100`, `gfx1101`, `gfx1150`, `gfx1151`, `gfx1152`, `gfx1200`, and
  `gfx1201` through the retired ROCm 7.2.1 profile, which also required AMD Software
  26.2.2 or newer. Those GPUs now fall back to the next eligible accelerator or to
  the CPU.
- **The managed environment is rebuilt on first launch.** The launchers recreate a
  `.venv` that does not run Python 3.13, so the first launch after upgrading rebuilds
  it with Python 3.13 and reinstalls every dependency, including PyTorch. Only the
  EMAP-SSN and EMAP-SSN Tools launchers do this. The MCP server, the jobs and Viewers
  it starts, and scripts run directly with the `.venv` Python, such as
  `src/tools/*.py` or `src/Layout_Cache_Generator.py`, use `.venv` as they find it:
  until a launcher has rebuilt it, they run the new code on the v0.2.0 Python 3.12
  environment, where ESMC embedding jobs, for example, fail to import `EsmcModel`.
  After upgrading, close EMAP-SSN windows and MCP clients (the rebuild replaces the
  environment they run from), launch EMAP-SSN or EMAP-SSN Tools once, and then
  restart the MCP clients.
- **Restart Viewers that were running during the upgrade.** The MCP server now asks a
  Viewer whether it supports alignment snapshots and node projection. A Viewer started
  before the upgrade does not, so `get_summary` with `include_alignment`,
  `get_residue_distribution`, and `query_nodes` with `fields` fail with "upgrade and
  restart the Viewer" until it is restarted. Plain `get_summary` and `query_nodes`
  calls fail too, typically with "Extra inputs are not permitted" rather than that
  message, because the upgraded server sends arguments the old Viewer does not
  accept. Since `get_summary` creates the snapshots that the other inspection actions
  read, restart such a Viewer before inspecting it over MCP.

### Added

- 2D or 3D layout generation for the physics and UMAP engines through
  `LAYOUT_DIMENSIONS` (`2` or `3`, default `2`) in layout settings documents,
  `Layout_Cache_Generator.py`, and `emapssn_pipeline(action="start_layout_job")`.
  3D caches are stored in separate folders with a `_3D` suffix, record `physics_3d`
  or `umap_3d` as their layout mode, and are validated as `(node_count, 3)`
  coordinates. The desktop Viewer, and Viewer settings validated through MCP, reject
  a 3D cache with a message that names it as one and points to its 2D cache or
  `LAYOUT_DIMENSIONS=2`. 3D physics layouts place disconnected components on
  concentric spherical shells around the largest one instead of packing them into a
  grid, so `PACKING_GEOMETRY` has no effect there; `PACKING_GRID_SIZE` sets the gap
  between shells and `PACKING_PADDING` the clearance around each component.
- `LAYOUT_SEED` (a non-negative integer, default `42`) seeds the physics and UMAP
  engines; `null` opts out of seeding. UMAP already used a fixed seed of 42 in v0.2.0,
  and physics layouts are now seeded too. With a fixed seed, UMAP layouts and CPU
  physics layouts reproduce exactly. GPU physics runs can still differ slightly between
  runs, because forces are accumulated with atomic additions, and
  `LAYOUT_DEVICE_SELECTION` `auto` picks devices from a timing benchmark, so select the
  CPU when a physics layout must reproduce exactly. Layout settings documents written
  before these keys existed load with the defaults.
- FASTA-derived node metadata in every new layout cache: `Length`, `kDa`, `pI`
  (Bjellqvist), and `GRAVY`, computed with vectorized residue counting and
  ambiguity-code handling and checked against Biopython's ProtParam in the test
  suite.
- Negative bounds in metadata ranges, written in parentheses as negative alignment
  positions are: `{GRAVY=(-1)-0}` or `{GRAVY=(-1.5)-(-0.5)}`. Ranges were split at
  the first `-`, so none could have a negative lower bound, which the new, usually
  negative `GRAVY` column needs. An unparenthesised negative bound such as
  `{GRAVY=-1-0}` is rejected with an explanation instead of "not numeric". A single
  value may be parenthesised too (`{GRAVY>=(-1)}`), and `{GRAVY>=-1}` works as
  before.
- Alignment snapshots for agents. `emapssn_viewer_data(action="get_summary")` with
  `include_alignment=true` freezes the displayed alignment into the snapshot; the new
  `get_residue_distribution` action returns paged per-position residue counts with
  cluster and group cross-tabulations; `create_subset` accepts residue predicates
  against such snapshots; and `query_nodes` gains a `fields` row projection.
- On Windows, the integrated Intel Arc 130V/140V and 130T/140T GPUs are recognized from
  adapter names such as "Intel(R) Arc(TM) 140V GPU" and are eligible for the XPU
  backend. Arc B370/B390, which v0.2.0 already accepted as discrete GPUs, are now
  classified as integrated, so a discrete GPU ranks ahead of them; "Arc Pro" B370/B390
  names are recognized too, and on Linux every B370/B390 name follows the Panther Lake
  rule (Ubuntu 25.10 or 26.04). Linux already recognized 130V/140V; the Arrow Lake
  "Arc Pro 130T/140T" name is still not matched there.
- The optional `opt_vr` Git submodule (EMAP-SSN-VR), a VR viewer for 3D caches for
  Windows users with a supported NVIDIA or AMD GPU and a VR headset with an OpenXR
  runtime such as SteamVR. EMAP-SSN does not need it, and GitHub source archives do
  not include it. Fetch it with `git clone --recurse-submodules`, or with
  `git submodule update --init opt_vr` in an existing clone. This release records
  the EMAP-SSN-VR commit that works with it, and `git pull --recurse-submodules`
  keeps the two in step. The VR client itself is not in git: `opt_vr\install_vr.bat`,
  or the first **Save & Run** in the VR Configuration window, downloads client 1.0.0,
  a Godot 4.7.2 build published as a release asset of EMAP-SSN-VR, and checks it
  against the SHA-256 pinned in `opt_vr/player_release.json`. Like the desktop
  Viewer, the VR viewer opens a cache only with the FASTA and network it was built
  from, and uses the cache's own edge filter and analysis settings. See
  `opt_vr/README.md`.
- `src/esm_runtime_requirements.txt` declaring ESM's runtime dependencies, installed
  after the accelerator-specific PyTorch build. It replaces the generated file that
  lived beside the bundled wheels, and the installer rejects it if it ever names
  `torch` or `transformers`.
- Linux AMD GPUs whose product name is not in the pinned model-to-GFX snapshot now
  resolve their target from `rocm_agent_enumerator`/`rocminfo` instead of being
  reported as unsupported. This makes AMD Instinct parts (`gfx908`, `gfx90a`,
  `gfx942`, `gfx950`), RDNA1/RDNA2 GPUs including APU graphics
  (`gfx1010`–`gfx1012`, `gfx1031`–`gfx1036`), and `gfx1153` eligible on Linux for
  the first time, and also covers cards of already-supported targets whose `lspci`
  names the snapshot does not match. The reported target is accepted only when
  exactly one supported target is present — with several distinct supported targets
  there is no reliable way to attribute one to a specific adapter — and only when the
  pinned PyTorch 2.12.0 ROCm 7.14 wheel declares a `device-<target>` extra for it.
  Windows target resolution is unchanged and still relies on the name snapshot,
  because it has no equivalent runtime source.

### Changed

- Upgraded `esm` 3.3.0 → 3.4.1.post1 (MIT, from PyPI) and `transformers`
  `4.57.6+biohub.3a8956f` → 5.17.0 (Apache-2.0, upstream). This also moves
  `huggingface-hub` from 0.x to 1.x and `tokenizers` to 0.23 (1.32.0 and 0.23.2 at
  release time). Neither is pinned, so installs resolve them within Transformers
  5.17.0's ranges.
- Pinned `torch` to 2.12.0 across every backend, including both ROCm platforms, so a
  single PyTorch version now covers the whole candidate ladder. For the CPU, CUDA,
  XPU, MPS, and Linux ROCm 7.2 backends this is a step down from v0.2.0's 2.12.1,
  because AMD's channel does not publish 2.12.1; the retired Linux ROCm 6.4 and
  Windows ROCm 7.2.1 profiles used 2.9.1, and Windows ROCm 7.14 moves from
  `2.12.0+rocm7.14.0` to `2.12.0+rocm7.14.1`.
- Consolidated the four ROCm profiles (Windows 7.14 and 7.2.1, Linux 7.2 and 6.4)
  into one ROCm 7.14 backend serving Windows and Linux from
  `repo.amd.com/rocm/whl-multi-arch`, so Linux ROCm wheels now come from AMD's
  channel instead of `download.pytorch.org`. ROCm 7.14 covers every GFX target the
  retired profiles did, so no GPU architecture loses support, but on Windows it
  requires Windows 11 25H2 (see Upgrade Notes). v0.2.0 retried a second ROCm profile
  on some targets; a ROCm 7.14 install or validation failure now falls through to the
  next accelerator or the CPU. The retired Windows ROCm 7.2.1 profile was limited by
  AMD to Python 3.12, which is why it could not move to the Python 3.13 environment.
- Raised the managed virtual environment from Python 3.12 to 3.13 in all four
  launchers. Every probe that decides whether to reuse `.venv` checks the version, so
  an existing `.venv` on another Python version, such as one kept from v0.2.0, is
  recreated instead of reused.
- `Generate_Embeddings.py` now hides Transformers weight-load reports whose only
  entries are UNEXPECTED. Loading into encoder-only classes leaves pretraining heads
  unused, such as ProtBERT's `cls.*` tensors and Ankh's `lm_head.weight`, and
  Transformers 5 lists them as UNEXPECTED, which is routine noise; Transformers drops
  the T5 decoder itself without reporting it. Reports with a MISSING, MISMATCH, or
  CONVERSION entry are still shown: a missing tensor is randomly initialized, and for
  the other two Transformers raises an error that points to the report. ESM-2 no
  longer builds the `EsmModel` pooler, which its checkpoints lack and its embeddings
  never used, so its loads no longer print a MISSING report; ESM-2 embeddings are
  unchanged. Other Transformers warnings are unaffected.
- Upgraded `numba` to 0.67.0, which raised its ceiling to `numpy<2.6` and unblocked
  `numpy` 2.5.3. Also bumped PySide6 6.11.2, biopython 1.88, matplotlib 3.11.2,
  mcp 2.2.0, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, sentencepiece 0.2.2,
  tqdm 4.70.1, and vispy 0.17.0.
- Backend state schema raised to 6 and `COMPATIBILITY_REVISION` to 7. The
  bundled-wheel checksum fields are gone and the ROCm profiles are collapsed, so
  schema-5 state is not comparable. The launchers' readiness check rejects it, and a
  v0.2.0 environment is recreated on first launch anyway because it runs Python 3.12.
  The state's fingerprints of `src/requirements.txt` and
  `src/esm_runtime_requirements.txt` now hash the requirement lines pip reads instead
  of the files' bytes, so editing a comment or a blank line, or checking the files
  out with other line endings, no longer forces a reinstall; any change to a
  requirement line still does.
- `validate_package_consistency` now parses `uv pip check` output instead of reading
  only its exit code. ESM is installed with `--no-deps` against a newer torch and
  Transformers than it declares, and its runtime requirements deliberately omit the
  ESMFold2-only `cuequivariance` kernels, so the check always reports those
  deviations: two version findings on every platform, plus two missing-package
  findings on Linux x86_64. A version finding is accepted only when the installed
  version is the one pinned here — for torch, its base version, whatever accelerator
  suffix such as `+cu132` follows it — and a missing package only when it is one of
  the documented omissions; any other finding still fails. An unaccounted-for or
  unparseable report fails closed.
- The ESM import smoke test now exercises `esm.models.esmc` and core Transformers
  classes (`AutoModel`, `AutoTokenizer`, `T5EncoderModel`), rather than the
  `ESMCModel` and `ESMFold2Model` classes that v0.2.0 imported from the Biohub fork's
  `transformers.models.esmc` and `transformers.models.esmfold2` modules.
- Every local pLM plugin requests float32 weights explicitly, because Transformers 5
  otherwise loads a checkpoint's stored dtype. Compute precision therefore belongs to
  the plugin rather than to checkpoint metadata, and `SAVING_MODE` is purely an
  on-disk storage choice. ESMC loads the bare `EsmcModel` and `EsmcTokenizer`
  instead of the deprecated `ESMC` wrapper, which forced bfloat16 on non-CPU
  devices, and ProstT5 loads `Rostlab/ProstT5` instead of `Rostlab/ProstT5_fp16`.
  `esmc_6b` still runs on Biohub's servers at their precision.
- The Ankh adapter compensates for Transformers 5's T5 tokenizer, which is built on
  the Rust `tokenizers` backend with a Metaspace pre-tokenizer that always prepends
  the sentencepiece word-start marker. Ankh's vocabulary contains bare amino acids and
  no such marker, so every unspaced sequence would gain a leading `<unk>` token and
  one extra embedding row. The adapter rebuilds the pre-tokenizer without the prefix
  only when a probe shows the extra token, and verifies residue alignment when the
  model loads. The ProtBERT and ESM-2 adapters were checked and are unaffected.
- `protobuf` is now a pinned dependency. ProstT5 ships only `spiece.model` and no
  `tokenizer.json`, so Transformers 5 converts it to the Rust `tokenizers` backend on
  first load, and that conversion parses the model with protobuf. Without protobuf,
  Transformers falls back to a TikToken extractor and fails with a misleading
  "`tiktoken` is required to read a `tiktoken` file" error. ProstT5's token alignment
  was verified and needed no change.
- The `esmfold` worker passes ESM's 0–1 pLDDT values to `to_pdb_string()` unchanged,
  because ESM 3.4 scales them itself. Multiplying by 100 first, as the worker did for
  ESM 3.3.0, would overflow the six-column PDB B-factor field.
- `color` and `spectrum` prefer a valid selection expression over a color or colormap
  name, so residue tokens such as `C53` select residues; `color` scales take the
  trailing-`x` form, so `x2` selects residues too. Command help, Viewer command
  metadata, and the agent's system prompt describe the new syntax.
- Initial node metadata is created during layout-cache generation instead of when the
  Viewer opens a cache; the Viewer only orders what the cache provides. As in v0.2.0,
  a stored metadata group without `Length` counts as an intentional column deletion.
- Integral float metadata values display and export without trailing decimals, and
  the `meta show` HUD readout uses the same formatter.
- `Length` is stored and loaded as a floating-point column, like `kDa`, `pI`, and
  `GRAVY`; v0.2.0 converted it to integers when it generated or loaded the column.
  The Viewer still displays and exports whole numbers, but MCP `query_nodes` and
  `read_value` now return values such as `350.0` instead of `350`.
- Viewer settings no longer reject an MSA that lacks the configured
  `ALIGNMENT_REFERENCE`. The requested reference and offset are preserved, and the
  Viewer falls back to occupancy-based numbering and logs a warning at startup.
- On a complete network, the Embedding MSA tool's imputed-consensus switch now shows
  OFF and its tooltip says "Not applicable"; v0.2.0 only disabled the switch in its
  previous state. The preference chosen for incomplete networks is kept, restored
  when an incomplete network is selected again, and saved.
- MCP pipeline jobs submitted with individual `parameters` take an omitted or blank
  directory from the project's saved Tools directories (`DIRECTORIES` in
  `tools_settings.json`), falling back to the built-in relative default only when
  none is saved; v0.2.0 always used the built-in defaults. `settings_document` and
  `settings_path` submissions still use the built-in defaults. `start_layout_job`
  reads the saved `SAVED_LAYOUT_DIR` from `viewer_settings.json` on every call;
  v0.2.0 read it once, when the MCP server first loaded the Viewer configuration.
- Neighbor lookups for the current selection use a cached CSR adjacency index instead
  of scanning every edge. Out-of-range and duplicate edges are dropped when the index
  is built, self-edges never change the result, and the index is rebuilt only when the
  edge list changes. The Viewer also skips edge-geometry uploads unless node positions,
  visibility, or the thresholded edge set change.
- Metadata upload, download, formatting, state synchronization, and column deletion
  moved from `web_ui/meta_backend.py` into `src/Metadata_Core.py`, which does not
  depend on PySide6 and is shared with the VR frontend. `meta_backend` re-exports the
  functions, so existing imports keep working.
- `load_alignment_smart` returns the loader, or `None` when the file is rejected,
  instead of a `(loader, is_sparse)` pair. `resolve_selected_cache` returns the cache
  path instead of a `(cache_path, reference)` pair (see Removed), honours
  `LAYOUT_DIMENSIONS` by naming the `_3D` folder for a 3D layout, and takes a
  `layout_dimensions` keyword that overrides the setting. Desktop settings carry no
  such key, so the desktop Viewer still resolves the 2D folder.
- The README and `docs/mcp_settings.md` document the three MCP entry points, Viewer
  sessions and actions, the layout-cache workflow, and alignment snapshots.
- The MCP server reports version `0.11.0` to clients (previously `0.10.0`) to reflect
  the additive interface changes in this release.

### Fixed

- `logo` numbered positions differently from `query` when no reference was active.
  It counted the non-gap residues of the first MSA row instead of using the occupancy
  labels, so after any gap in that row the same position number plotted a different
  alignment column, and the axis label named the first sequence or the inactive
  reference as the numbering basis. `logo` now uses the alignment's label mapping in
  both modes, and its axis reads "occupancy numbering" in occupancy mode.
- `print svg` did not match the displayed network. It drew every edge between visible
  nodes, including edges below the similarity threshold and, in UMAP mode, edges not
  attached to the selection, and it drew edges and node outlines in black. It now
  applies the same edge filter as the screen and the configured `EDGE_COLOR` and
  `NODE_BOUNDARY_COLOR`.
- The alignment reference was resolved with different rules in different places. A
  wildcard target matched a network header but was then looked up literally in the
  MSA, so `reference WP_01*` left the reference inactive with a message claiming the
  sequence was absent. When several headers matched, the warning and the success
  message could name a different sequence from the one the alignment anchored on. A
  reference whose sequence the MSA lacks was anchored on another row whose header
  contained it or was contained in it, so `reference P12_kinase` could number
  positions against `P1`. The `reference` command and the `ALIGNMENT_REFERENCE`
  setting now resolve the target once against the network headers, and the alignment
  anchors only on that exact header: wildcard targets work, the success message names
  the sequence the alignment anchored on, and a reference missing from the MSA stays
  inactive, with the MSA in pure occupancy mode. That mode no longer keeps the columns
  of the `ALIGNMENT_REFERENCE` sequence after `reference` selects a sequence the MSA
  lacks. A bare `reference` marks an unresolved reference as inactive. An exact header
  takes priority, then a leading identifier, then substring matches, so
  `reference WP_0123.1` selects `WP_0123.1_protein_A` instead of the first header
  containing it, such as `WP_0123.10_protein_B`, and wildcards such as `*.1` can
  match that identifier. The Config GUI's consistency check applies the same rules:
  it names the header the reference resolves to, any headers that tie with it, and
  whether the MSA lacks it, instead of counting every header that contains the text,
  and it no longer reports a wildcard as missing.
- Headless Viewer launches through MCP on Windows opened a stray terminal window.
  The launcher started the venv's `python.exe` redirector without any console, so
  Windows gave the interpreter it starts a new, visible one; when that process ended
  early, Windows Terminal left an error message on screen. Headless launches now use
  a hidden console, and normal-mode launches still open their terminal on purpose.
- Sanitize Sequences silently dropped the first record of a FASTA file saved with a
  UTF-8 byte-order mark, as Windows Notepad can write. Its private reader decoded the
  BOM into the first line, which then no longer started with `>`. The tool now uses
  the shared FASTA reader, which strips the BOM, and still writes UTF-8 without a BOM,
  including in overwrite mode.
- The Configuration window refreshes the normalization-mode options before applying
  the default even while its signals are blocked, so the list stays in sync with the
  selected score mode and no longer silently rejects `alignment_length`.
- `label` misread thresholds written as small percentages: `0.5%` counted as 50% and
  `1%` as 100%, because the percent sign was dropped and only values above 1 were
  divided by 100. A trailing `%` now always means percent, as in `logo` and `query`;
  bare values keep their meaning (`0.4` and `40` both mean 40%), and `nan` and `inf`
  are rejected.
- The Viewer command catalog (`emapssn_viewer_data(action="get_command_catalog")`)
  cut off help text written as an f-string: `meta`'s entry stopped at "the metadata
  directory:", and its syntax list lacked `download`, `delete`, and `help`. Help
  text is now read whole, with placeholders shown as `<meta_dir>`, and alternative
  usage lines written `or: …`, such as `label`'s keyword form, are catalogued too.
- The in-app agent's system prompt misdescribed parts of the Viewer's grammar, so the
  agent could generate commands that fail. It gave header-list files as `@[FILE]@`,
  which the parser reads as a file literally named `[FILE]`, instead of `@FILE@`; it
  told the agent to rely on an `ACTIVE EMAP-SSN VIEWER STATE` block that the agent is
  never sent, instead of the `ACTIVE VIEWER SNAPSHOT` it receives; and it did not rule
  out `subcluster #cluster_N#`, which `subcluster` rejects. The prompt now matches
  the parser.
- `select <EXPRESSION> save` raised an internal error, because `save` was accepted as
  a mode after an expression. `save` is only valid as `select save <FILENAME>`, and
  other placements now fail with a message saying so.
- `select save <FILENAME>.fasta` re-read `NODE_FASTA_FILE` and looked the selected
  nodes up by its raw headers, so records whose header sanitizing changes, such as
  ones with spaces, brackets or a `/`, were reported missing from the source FASTA and
  left out of the file. It now saves the sanitized sequences the Viewer loaded and
  checked against the layout cache, under the same canonical headers as
  `select save <FILENAME>.txt` and `export`, in both the desktop and VR viewers.
- `spectrum` treated infinite metadata values as numbers. A single `inf` stretched the
  color range to infinity, so every other node took the lowest color and the
  infinite node became transparent. Infinite values are now colored gray and counted
  as invalid, like NaN.
- `color` accepted negative, NaN, and infinite scales (in v0.2.0's syntax, `x-2` or
  `xnan`). Such scales, for example `-2x`, `nanx`, or `1e400x`, are now rejected
  before any change is applied; `0x` remains valid.
- `subcluster clear` removed custom groups whose names only resembled generated ones,
  such as `subcluster_0_2` or `subcluster_001_2`, although `group` accepts those names,
  and `subcluster cluster_2` removed custom groups such as `subcluster_2_002` before
  writing its labels. Both now remove only generated labels, whose two IDs are
  positive integers without leading zeros: the names `group` reserves.
- `emapssn_pipeline(action="start_layout_job")` silently ignored `parameters` keys
  that are not layout fields, so a misspelled key such as `SPRNG_K` left the default
  in place. Such keys are now rejected with the closest valid field name, and keys
  that belong to other arguments (`node_fasta_file`, `input_hdf5`, `cache_filename`,
  `directories`) are rejected with a pointer to that argument.

### Removed

- ESM-2 `esm2_t36_3b` and `esm2_t48_15b` from model support, execution-mode mapping,
  and Hugging Face ID resolution.
- The prefix `x<number>` scale syntax of `color`.
- `src/resources/agent/agent_config.json`, an unused single-model agent configuration
  that no code had read since the 2026-07-07 restructure. The agent reads per-user
  model cards from `model_card.json`.
- The Windows ROCm 7.2.1 detection path, its GFX target set, and the AMD Software
  26.2.2 version gate. The historical `rocm721`, `rocm72`, `rocm714` and `rocm64`
  profile names are gone entirely: the detector reports a single `rocm` profile,
  which the installer consumes directly. State saved by an earlier install is
  rebuilt (see the backend state entry under Changed).
- `_windows_amd_software_version()`, which ran a PowerShell registry query on every
  detection. It existed only to gate the retired ROCm 7.2.1 profile, so nothing
  reads it now, and the detection report no longer includes
  `os.amd_software_version`.
- `src/resources/wheels/` and its contents: the ESM 3.3.0 wheel, the
  `4.57.6+biohub.3a8956f` Transformers wheel and its packaging patch, the wheel
  manifest, the generated ESM runtime requirements, and the adjacent license texts.
  Git history retains them. The upstream fork repository the Transformers wheel was
  built from is no longer reachable, so its documented rebuild recipe could not be
  followed regardless.
- The bundled-wheel verification path in `src/Install_Dependencies.py`
  (`verify_bundled_artifacts`, `verify_esm_wheel`, `verify_transformers_wheel`,
  `esm_runtime_requirements_from_wheel`, `_bundled_paths`, and the SHA-256
  constants).
- The non-sparse alignment code path. Every MSA, FASTA or HDF5, loads as a sparse
  matrix, so `Alignment_Manager`'s branches for Bio alignments,
  `get_valid_columns_legacy`, `get_ref_anchored_mapping_legacy`, the per-row
  residue-predicate fallback in `Command_Engine`, and the matching fallbacks in
  `query`, `logo`, and `label` could never run. The `reference` command's second
  search over MSA rows is gone too: the alignment holds only rows whose headers are
  network headers.
- `Alignment_Manager.calculate_frequencies`, the module-level `calculate_frequencies`
  it wrapped, and `SparseAlignmentLoader.get_frequencies`, which only that function
  called. Nothing in the application computed statistics through them; `label`,
  `logo`, `query`, and MCP residue distributions each count residues themselves.
- The reference lookup in `resolve_selected_cache`. It searched the MSA file for the
  first header containing `ALIGNMENT_REFERENCE`, a rule the alignment does not use,
  on every call, including each `save`, Mol* session save or load, and agent history
  lookup. Only the Viewer's startup kept the result, as the header a bare `reference`
  reported, which could differ from the one the alignment anchored on; every other
  caller discarded it. `Alignment_Manager.resolved_ref_full` now holds the header
  the alignment anchored on.

## [0.2.0] - 2026-09-09

### Highlights

- **Model Context Protocol (MCP) Server**: Integrated local stdio MCP server (`src/EMAPSSN_MCP_Server.py`) exposing the full suite of 14 pipeline programs, hardware compute capability discovery, structural file inspection, and interactive Viewer session management to external AI agents and IDEs.
- **UMAP 2D Manifold Layout**: Added alternative non-linear manifold projection powered by UMAP over pairwise distance matrices, providing an alternative to 3D spring-electrical force-directed physics.
- **Multimodal AI Agent & Web Plugins**: Embedded conversational AI agent, Mol* 3D structure viewer (ESMFold/PDB), and Tabulator interactive metadata explorer with bidirectional selection synchronization directly in the 3D Viewer.
- **Headless Layout Generation & Versioned Schema**: Introduced `src/Layout_Cache_Generator.py` for decoupled HPC/batch layout calculation and adopted the v2 execution settings schema with strict provenance tracking.
- **Hardware Acceleration & Precision**: Added two-stage BF16 precision validation, adaptive tile planning for Apple Silicon MPS, and unified GPU/CPU physics force balances.
- **Documentation Suite**: Added comprehensive offline-ready Quick Start Manual (`docs/quickstart.html`), updated Viewer Command Reference (`docs/list_of_commands.html`), and detailed guides for MCP settings and web plugin development.

### Added

- Stdio MCP server supporting Claude Code, Codex, Antigravity, and VS Code with bounded queues, live log streaming, settings validation, and job cancellation.
- UMAP 2D manifold layout engine (`src/Layout_Engine_UMAP.py`) with configurable $k$-nearest neighbors and minimum distance parameters.
- Web plugin discovery subsystem (`src/web_ui/plugins/`) hosting `meta.py` (metadata table), `esmfold.py` (Mol* 3D structure viewer), and `agent.py` (multimodal AI copilot).
- Headless layout cache runner (`src/Layout_Cache_Generator.py`) for background and cluster execution without Qt/GUI initialization.
- Grouped amino acid query syntax in the HUD console (`#polar#`, `#charged#`, `#aromatic#`, etc.).
- Base directory aliases (`$input_file$`, `$cache_file$`, `$analysis_result$`) for portable workspace configurations across systems.
- Comprehensive Quick Start Manual (`docs/quickstart.html`) with end-to-end workflow diagrams, GUI tours, and a 5-step tutorial.

### Changed

- Renamed project to **EMAP-SSN** (*Embedding- and Multiple-Alignment-integrated Protein Sequence Similarity Network Platform*) and standardized core entrypoints to PascalCase (`EMAPSSN_Viewer.py`, `EMAPSSN_Tools.py`, `EMAPSSN_Config.py`).
- Refactored internal architecture into modular packages: `desktop/`, `utilities/`, `commands/`, `mcp_server/`, and `tools/tool_helpers/`.
- Updated default embedding calculation precision to `float32` for increased numerical stability.
- Mandatory layout cache generation before opening the interactive Viewer to guarantee reproducibility and prevent redundant layout simulations.
- Replaced the deprecated Monte Carlo layout engine with the unified and validated force-directed physics solver.
- Enhanced PNG figure export with fixed padding margins and border trimming.

### Fixed

- Fixed sparse resume pair counting during long-running pairwise embedding alignments.
- Fixed GPU and CPU Coulomb repulsion balance in force-directed simulation.
- Fixed sidebar active-link highlighting on direct click in the HTML documentation.
- Hardened temporary session and lockfile handling on Windows hosts.

## [0.1.0] - 2026-09-02

### Highlights

- First public release of the EMAP-SSN platform for constructing, visualizing,
  and analyzing traditional and embedding-based protein sequence similarity
  networks.
- Integrated BLAST and protein-language-model workflows, embedding-based
  pairwise and multiple-sequence alignment, network clustering, residue-level
  conservation analysis, and structure-prediction/inspection utilities.
- Interactive PySide6 and VisPy desktop interfaces for configuration, tools,
  network exploration, command-driven analysis, and publication-quality
  exports.
- Managed Python 3.12 environments with conditional CPU, NVIDIA CUDA, AMD ROCm,
  Intel XPU, and Apple MPS backend selection and runtime validation.
- Windows, Linux, and Apple Silicon macOS launchers with platform-dependent
  compatibility documented in the README.
- Apache-2.0 project licensing, third-party component inventory, bundled-wheel
  provenance, and explicit acknowledgement for separately licensed model
  weights.

### Distribution

- Distributed as GitHub-generated source archives. Native executables and model
  weights are not included.

[Unreleased]: https://github.com/Xuebin-Feng/EMAP-SSN/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Xuebin-Feng/EMAP-SSN/releases/tag/v0.3.0
[0.2.0]: https://github.com/Xuebin-Feng/EMAP-SSN/releases/tag/v0.2.0
[0.1.0]: https://github.com/Xuebin-Feng/EMAP-SSN/releases/tag/v0.1.0
