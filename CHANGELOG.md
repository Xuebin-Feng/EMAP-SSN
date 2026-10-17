# Changelog

All notable changes to EMAP-SSN are documented in this file. The project uses
[Semantic Versioning](https://semver.org/); interfaces and persisted formats may
still change before version 1.0.0.

## [Unreleased]

### Upgrade Notes

- Settings that select an embedding are not readable by v0.4.0. Going back, it refuses an
  Inputs & Outputs profile that names an embedding file, and that tab falls back to its
  defaults, and it refuses a Viewer settings document exported with an `embedding`
  section. Profiles and documents saved by v0.4.0 keep working.
- VR Config and the VR viewer resolve default and relative folders against the project
  root instead of `opt_vr`. A VR setup that relied on the old defaults or on relative
  paths now uses the project's `Input_Files`, `Embeddings` and `Analysis_Results`, and
  `Cache_Files_VR` for its cache. Profiles, layouts and history saved in
  `opt_vr\Cache_Files` are no longer read: copy them to `Cache_Files_VR`, or point the
  Cache File Directory at the old folder. Move inputs from `opt_vr\Input_Files` to
  `Input_Files`. Absolute folders are unaffected.

### Added

- **Optional embedding input.** Config and VR Config can select a protein language model
  embedding file (**Embedding Input**, listed from the new **Embedding Directory**, by
  default the project's `Embeddings` folder), for commands that use each sequence's
  embedding. No command uses it yet. The Embedding Directory is a fourth base directory
  on the Directories tab, beside the input, cache and analysis-results ones, and other
  folders can name it as `$embedding$`. Only the file's
  headers, sequences and model are read at launch; a command reads per-residue rows or
  pooled vectors when it needs them. A node is matched by its exact header and sequence,
  partial coverage gives a warning as for the MSA, and an incomplete file is refused at
  launch. The Consistency Check reports the embedding's coverage, the sequences that
  differ from the FASTA and its model, with a note when the network was built with
  another model. Viewer settings documents gain an optional `embedding` section and
  `EMBED_DIR`; documents exported before still validate.
- **Commands name the inputs they need.** Each command's catalog entry lists the optional
  inputs it cannot run without (`requires`: `alignment` for `label`, `logo` and `query`),
  and MCP `get_command_catalog` returns it. Commands that will use the embedding share
  one check, with one message for a missing, failed or non-matching embedding file.
  Viewer snapshots report the embedding's model, width and node coverage (capability
  `embedding_input_v1`), never its vectors.
- **Session on the Viewer's info card.** The ⓘ card at the start of the threshold slider,
  now titled "Session and controls", lists the number of nodes and edges, with how many
  are visible and how many edges show at the current threshold, and the sequence set,
  network, alignment, embedding and layout cache the Viewer opened, each with an Open
  Folder button. The values follow the threshold and commands while the card is open.
  Below the mouse and keyboard controls, a link opens the list of commands in the
  Viewer's language, or in English if that language has no translated copy.

### Changed

- VR Config and the VR viewer use the same default folders as the main program, at the
  project root, with one exception: the cache folder defaults to `Cache_Files_VR`, so
  VR's saved configs, 3D layouts and history stay apart from the main program's.
  `viewer_settings_vr.json` and the VR client stay in `opt_vr`. VR Config and its viewer
  now start in the project root, so relative paths typed in commands resolve as on the
  desktop.

- Config, Tools and VR Config start faster. The launcher's check imported the ESM and
  Transformers packages at every launch, which took about 6.5 s, or about 20 s when the
  programs had not been opened for a while (Windows Defender scans every file the import
  reads). A passing import test is now recorded in `.venv/ssn_esm_stack.json` with a
  fingerprint of the interpreter and the installed packages, and skipped until a package,
  the interpreter or the test changes; deleting the file runs it again. A warm Config
  launch takes about 7 s instead of 15 s.
- Windows hardware detection at launch runs its four PowerShell queries at the same time
  and asks for the processor name alone, about 0.55 s instead of 2.5 s, with the same
  result.
- The Consistency Check in Config and VR Config shows each check as one row of a compact
  table (a status icon, the check, the counts and the verdict) instead of three or four
  lines. Whatever a check found wrong is listed below the table: missing headers,
  sequences that differ, a different embedding model or an unreadable file. A long
  report is no longer cut off at the bottom of its panel, and a long header no longer
  widens the window.
- The Quick Start manual (`docs/quickstart.html`) has the look of the windows and the
  Viewer's web pages: their colours in light and dark, Noto Sans, cards, tabs, buttons and
  Lucide icons in place of emoji. A top bar switches between the Quick Start and the
  Command Reference, and the sidebar folds into a drawer on a phone. The style lives in
  `docs/assets/docs.css` and `docs.js`, which keep the theme choice for both pages; a test
  fails if the colours there drift from the windows' theme. Text and anchors are unchanged.
- The Viewer's info card stays open while you click, drag or scroll on the network, and
  those actions work as usual. Only its × or the ⓘ button closes it. Before, any click
  closed the card, and that click did nothing else.

### Fixed

- A long dropdown list in Config, Tools or VR Config scrolled by itself while the mouse
  rested on its partly visible last row, and a click after a drag to that row picked a
  row further down than the one pointed at. An open list now scrolls only with the mouse
  wheel, the scroll bar or the keyboard, and a click selects the row under the pointer.
- VR's `meta` command created the metadata folder at every use, even for `help`, `show`
  or `delete`, so a VR session left on the default folders, and the VR tests, left an
  empty `Input_Files/Meta_Data` behind. Like the desktop's, it now creates the folder
  only when it downloads metadata.

## [0.4.0] - 2026-10-16

### Highlights

- **A new look, theme colours and a second language.** Config, VR Config, Tools and the
  Viewer's pages share one look with line icons, and each window takes one of eight
  theme colours. Every window can switch to Simplified Chinese (简体中文, drafts awaiting
  review), and open windows follow a language chosen in another.
- **Much faster layouts of large networks.** GPU layout generation is about 50 times
  faster and fits large components, CPU layouts use several cores, spectral starts are
  solved far faster, and 2D and 3D layouts are packed more tightly. New layouts differ
  from v0.3.0's; saved caches open as before.
- **A responsive Viewer at tens of millions of edges.** The threshold slider, node
  dragging and box selection stay smooth on large networks, startup and many commands
  are faster, and the view width and threshold are boxes to type in.
- **A built-in benchmark.** **Run Benchmark** on the Tools window's Manual Tools tab, or
  MCP `start_benchmark`, times the program's heavy stages on 860 public UniProt
  sequences and writes a report to compare computers with.
- **MCP agents can plan and wait.** `list_tools` describes each pipeline tool's inputs,
  outputs and usual next steps, `wait_job` and `wait_session` wait for jobs and Viewer
  launches, and a slow Viewer launch is handed back instead of stopped (server 0.14.0).
- **Wider GPU support.** On Linux an AMD GPU's target comes from the kernel, so Instinct
  cards and unnamed Radeons get ROCm without ROCm tools; Ubuntu-based distributions get
  ROCm, and Instinct MI300 and MI350 also on the RHEL, Debian, SLES, Oracle and Rocky
  Linux releases AMD lists. Intel GPUs are recognized by PCI ID, and more Radeon RX
  7600-class cards get ROCm on Windows.
- **Safer by default.** Output names stay in their folders, Agent replies wait for
  **Run** before commands that write files, settings files are written atomically and
  never replaced when they cannot be read, and the Fold View makes no remote request.

### Upgrade Notes

- **Linux and macOS: run `git checkout -- src/bin/` before `git pull`.** v0.3.0's
  `install.sh` made the launcher scripts in `src/bin` executable and the repository now
  stores them that way, so in a checkout where it has run, the pull stops with "Your
  local changes to the following files would be overwritten by merge" until that command
  undoes the permission changes.
- **The first launch updates the managed environment.** Markdown and the ESM runtime
  dependencies are now pinned and `xlrd` 2.0.2 is new (for `.xls` metadata uploads), so
  the first launch of EMAP-SSN or EMAP-SSN Tools installs them; the MCP server and
  scripts run straight from `.venv` before any launcher run lack `xlrd`. A computer
  whose AMD or Intel GPU is now recognized also installs ROCm or XPU on that launch and
  keeps it if it passes validation.
- **Regenerated results differ from v0.3.0; saved caches do not.** New layouts differ
  (spectral starts, the GPU kernels, 2D and 3D packing, and `PACKING_GRID_SIZE` now
  defaults to 10 instead of 20, though a saved setting keeps its value), and so do
  `cluster` and `subcluster` results (MCL, Jaccard and Leiden), subcluster numbering,
  `logo` bits for gapped columns and the text of NA values in metadata imports. Layout
  caches already saved open as they were.
- **Output names, `print full`, `zoom` and `run` are stricter.** `select save`,
  `meta download`, `print` and `@file@` take plain file names only, and a name with a
  directory, a drive, a network share or a `:` stream suffix is refused. `print full`
  over 1000 megapixels, `zoom` outside the view's limits and a typed `run` script of
  more than 1000 commands are refused as well, and a typed `run` stops at its first
  failed or cancelled command.
- **Agent replies wait before commands that touch files.** `save`, `select save`,
  `print`, `export`, `logo`, `label` (except `label reset`), `esmfold`, `run`,
  `alignment` and `meta` uploads, downloads and file imports wait in the Agent page for
  **Run** or **Discard**, and a reply's `agent` lines are refused; commands typed in the
  Viewer or sent through MCP are unaffected.
- **MCP clients: `start_session` returns instead of raising.** When its wait ends while
  the Viewer is still loading, it returns `status: "starting"` with a `launch_id` and
  leaves the Viewer running, where v0.3.0 raised an error and stopped it; the default
  wait is 45 s instead of 30 s. The server version goes from 0.11.0 to 0.14.0, with
  every other change additive.
- **Close open Config, VR Config and Tools windows before updating.** Each copy of
  EMAP-SSN now opens its own windows, and the updated launchers do not recognize a
  window opened from the old version, so launching again from the same folder opens a
  second one beside it.

### Added

- **Theme colours.** Config, VR Config and Tools each take a theme colour of their own:
  Purple (the default), Blue, Teal, Green, Orange, Red, Pink, or Black, which has no
  colour: near-black buttons and switches, with grey sliders, selections and focus rings.
  A circle after the Language dropdown shows the window's colour; a click opens a palette
  of the eight, each named on hover, and a choice saves it for that window in
  `app_settings.json` and redraws the window in it at once, keeping everything it shows.
  Only the accents take the colour: the main button (such as **Save & Run**), switches
  that are on, focus rings, slider fills and selections; a light shade of it draws the
  2 px lines between a tab's sections, and a light tint of it fills the tab bar, the tips
  card and secondary buttons such as **Compute Network Statistics**. Each shade has a
  fixed depth, so no colour changes the contrast.
  Borders, icons, greyed-out fields, tooltips and status colours look the same in every
  theme colour. The one exception is Tools' Script Description: its text, borders and
  code backgrounds take a tint of the colour, and its collapsible sections' titles the
  colour itself, while its icons stay grey. The Viewer takes Config's colour when it
  starts, and its Agent and Meta Data pages take the Viewer's, in light and dark mode;
  the Fold View page takes no theme colour, since Mol* draws its own interface there.

- **MCP planning knowledge.** `emapssn_pipeline(action="list_tools")` describes each
  pipeline tool's stage, inputs (with the directory a relative name resolves in), output
  file names, what happens to an existing output, requirements and typical next tools,
  plus workflow recipes (embedding SSN, BLAST SSN, imported BLAST or DIAMOND results,
  adding sequences by injection, subsets, MSA, fragments) and path rules, so an agent
  can plan a multi-step calculation without reading tool source. `get_tool_schema`
  repeats the tool's entry and, for embedding models, reports which models an MCP job
  can use on this machine and what the user must do first for the others (store a Biohub
  token, accept a weights license). MCP server version 0.14.0 (was 0.11.0).
- MCP `emapssn_pipeline(action="wait_job")` waits up to 50 s for a job instead of
  polling. Job results list the files a job created, modified or deleted
  (`output_files`) and its latest stdout and stderr line (`latest_output`); layout
  jobs add a `result` summary of nodes, edges, effective threshold, clusters, isolated
  nodes and the largest cluster.
- MCP `emapssn_pipeline(action="network_statistics")` previews layout edge filters. It
  scores a network exactly as a layout job would for the chosen nodes, score mode and
  normalization, and reports the score distribution and, for each `TOP_EDGE_PERCENT`
  or threshold, the kept edges, clusters, isolated nodes and largest cluster (about
  20 s for 83.5 million pairs).
- MCP `emapssn_pipeline(action="get_layout_schema")` documents every layout setting
  (meaning, built-in default, accepted and Config GUI ranges) with edge-filter guidance,
  and `export_layout_settings` takes `node_fasta_file`, `input_hdf5` and layout
  `parameters`, so an agent inherits the saved physics settings without writing an
  overlay file. `inspect_file` reports FASTA length quantiles and duplicate sequences,
  and a network's gap penalties, precision or BLAST matrix. An unknown MCP argument
  error names the accepted arguments.
- MCP `emapssn_viewer_control(action="wait_session")` keeps waiting for a Viewer launch
  and connects when it is ready. `start_session` takes an optional `ready_timeout`
  (default 45 s, at most 600), and `close_session` accepts the launch's `launch_id`,
  which also stops a Viewer that is still loading.
- `emapssn_pipeline(action="inspect_file")` reports a layout cache's saved state —
  cluster count and noise nodes, groups, the last clustering parameters, and metadata
  columns — and its layout dimensions. Viewer `export_settings` reports the same state
  for the selected cache and for the folder's other caches, so a saved figure cache
  can be chosen deliberately.
- The ESM3 structure worker (`esmfold_worker.py`) gains `--skip-existing`, which keeps
  structures already on disk so an interrupted run can resume.
- `Parse_BLAST_Output.py` records DIAMOND provenance. DIAMOND output written with
  `--header verbose` declares its version and command line; the importer stores them,
  like a BLAST+ program line, in the network's `search_program`, `search_version` and
  `search_invocation` attributes and names the network `<name>_[DIAMOND]_EValue.h5`.
  `model_name` stays `BLAST`, so the Viewer still loads it as an E-value network.
  Layout-cache folders and `Network_Extraction.py` subsets made from it carry the same
  `[DIAMOND]` label.
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
- **Translation machinery, and Simplified Chinese as its first language.** The Config,
  Tools and VR Config windows and the Viewer can now take their text from Qt translation
  catalogs in `src/resources/languages/`.
  `python src/resources/languages/Update_Translations.py` collects the marked texts,
  including the Viewer's console messages, into the catalogs. It keeps each language's
  translations, compiles the catalogs and checks that translations keep their
  placeholders. With `--check` it changes nothing and fails while a catalog is out of
  date. `SSN_PSEUDO_TRANSLATION=1` starts a window in a test-only pseudo-language: every
  text from the catalog shows accented, longer and bracketed, so text never marked for
  translation, or cut off, stands out. See `src/resources/languages/README.md`.
  Simplified Chinese (简体中文) is the first language, shipped as a draft awaiting review:
  every text of the windows, the Viewer's canvas and the two browser pages, VR Config's
  own texts and the five tool help pages are drafted in Chinese with one glossary, and
  no native speaker has reviewed them yet. The drafts show until a reviewer accepts them
  in Qt Linguist, and the catalog update counts them apart ("0 of N texts translated; N
  more drafted, awaiting review."). The Language option below offers 简体中文, and "System
  default" picks it on a Simplified Chinese system. The browser views draw Chinese in
  the bundled Noto Sans SC too: a page served in Chinese, and the Tools help panel,
  carry its @font-face rules under one page family, 'EMAP-SSN Language', which their
  font stacks name after Noto Sans, as the windows' stacks name the language's font.
- **Language option.** Config, VR Config and Tools show a Language dropdown, marked by a
  globe icon, at the bottom right. It lists the system's language, English and every
  language with a compiled catalog, each named in itself. The choice is saved as
  `LANGUAGE` in a new `app_settings.json` in the project folder and applies to every
  window. Config, VR Config and Tools redraw in the chosen language at once, at the same
  size, position and tabs, keeping their splitter and scroll positions and every value
  entered, saved or not; a Network Statistics Report already computed shows again in the
  new language. Another of those windows that is open redraws the same way the next time
  it becomes the active window, as when it is clicked into, so a Config left open while
  Tools switched no longer stays in the old language while the Viewer it launches shows
  the new one. It waits while one of its dialogs or dropdown lists is open, and stays as
  it is while `app_settings.json` can't be read or holds no language yet. The Viewer and
  any dialog take the language when they next open. Simplified Chinese brings a font of
  its own, Noto Sans SC cut down to GB2312's 7,445 characters plus Latin and punctuation
  (4.4 MiB for Regular and Bold), which loads only while that language shows. A window
  redrawn into it draws bold text in its Bold, as one started in it does, also on Windows
  with Noto Sans SC installed, whose thin faces it otherwise drew.
- **A language is one folder.** Each language lives in `src/resources/languages/<code>/`
  (`zh_CN/`): its catalog `emapssn_<code>.ts` with the compiled `.qm`, an optional
  `language.json` and its translated tool help pages in `help/`. `language.json` gives
  the language's name where Qt's is ambiguous, its bundled font (for a script the core
  Noto faces lack) and how messages put together from sentences are punctuated, so no
  language is written into the code, and a language is added by adding its folder
  (`Update_Translations.py --add <code>` starts one). The font files stay in
  `src/resources/fonts/` with their licence and checksums. VR Config keeps its own
  texts' catalogs in opt_vr, in folders of the same names. Folder and file names count
  only as written, `zh_CN` and not `zh_cn`, on Windows and macOS as on Linux. The update
  command checks every `language.json` and reports a translated help page left beside
  the English pages.
- **Dark mode on the Meta Data page.** Its header has the Agent page's Dark Mode / Light
  Mode button, and its spreadsheet, filters and help take the Agent page's dark colours.
  The two pages share the choice: each opens in the theme chosen last on either, and
  while both are open, a choice made on one shows on the other at once. Until a choice
  is made, a page follows the system's dark setting. The Meta Data page sets its theme
  before it is drawn, so a dark page never shows light first.
- **Sanitize Sequences can skip its length histogram.** A Show Length Histogram switch
  (`SHOW_LENGTH_HISTOGRAM`, on by default) takes Enable Length Filter's place on the
  Tools card, and Enable Length Filter moves to the row of the Min and Max Seq Length it
  turns on. Turned off, a run ends without the histogram, and a run from a settings file
  (CLI or MCP) without the table it prints instead. MCP takes the setting too; a settings
  document without it keeps the histogram.
- **Every window is ready to translate.** The Config, Tools and VR Config windows, the
  Viewer's window and the launchers' failure dialog take their texts from the catalog.
  Config's are its labels, tips, dropdown choices, the statistics and consistency
  reports, messages and errors (its errors still print in English in the terminal).
  Tools' are its labels, tips, dropdown choices, card and tab titles, help-panel notes,
  messages, the external model license dialog and the names a screen reader speaks, with
  file and model names, devices, substitution matrices such as BLOSUM62, and float32
  staying as written. The Viewer's are its title, the sidebar's buttons and toggle, the
  metadata spreadsheet's file dialogs and the messages shown when a browser page opens
  (MCP clients still read the window's title in English). VR Config shows the language
  the Language dropdown chose, as the desktop Config does: the texts the two share come
  from the main catalog, and VR Config's own (the VR client's fields, tips and notes)
  from opt_vr's new catalog, `opt_vr/src/resources/languages/emapssn_vr.ts`, which
  `Update_Translations_VR.py` keeps beside the main one. When an application started
  from a desktop launcher fails and no terminal can open to show its log, the dialog
  that says so shows in the windows' language, or in English if that language can't
  load; the log stays English. Counted texts name their count with `%n` and mark their
  English plural endings, as in `%n file(s)`, so English shows "1 file" and "2 files"
  and a translation gives each plural form its language has. Some Config messages now
  read as correct English, such as "Created profile:" for one new profile and "The
  network lacks 3 sequences." Exporting metadata now picks CSV or Excel by the chosen
  filter's file pattern, which every language keeps. The update command refuses calls
  whose text or count Qt's `lupdate` would miss (arguments passed by name, and a counted
  `tr()`) and a text whose lines a backslash joins, which `lupdate` reads with a line
  break the code never shows; `install_translations` takes a window's extra catalogs,
  and `update_catalogs` can keep a catalog that leaves out another catalog's texts.
- **The Viewer's canvas text and the console messages of all 24 commands are ready to
  translate.** The HUD (View Width and Hidden Nodes) and the console-line messages of
  the Viewer, the command engine, background jobs, the metadata spreadsheet and the
  agent come from the catalog, the selection-expression errors included, and so do those
  of every command: agent, alignment, esmfold, offset, redo, reference, reset, run,
  save, undo and zoom (eleven); color, export, group, hide, meta, select and spectrum
  (seven); print, query, cluster and subcluster (four); and label and logo (two). Each
  takes its console messages, the first line of its help ("Usage: zoom <width>", the
  syntax staying English), its file dialogs and its error dialogs from the catalog.
  meta's HUD display shows "N/A" in the windows' language, and so do the errors of a
  file name that is a path or holds an unsupported character, of a position written
  wrongly (a bare negative position or an unknown label), of a frequency search, and of
  a full print capture that the window spoiled. The terminal, logs and MCP replies keep
  the English, so the commands that used to report what the console line shows as their
  failure no longer do: `label` no longer reads its console line back into a job's
  error, and a failed background job's error reaches the console line as itself, so it
  shows in the windows' language while MCP clients get the English. The scheduler's own
  refusals (an output file that exists or is reserved) are translated too. Counted
  messages read correctly for one, as in "Selected 1 node.", "Applied: 1 node (red)",
  "Done! Found 1 cluster via LEIDEN.", "Queried 1 position. Check terminal." and "Batch
  execution completed: 1 command run."; the metadata upload summary gives its counts as
  separate sentences, and the error for an invalid metadata predicate shows its example
  as '{Length>500}', no longer with doubled braces.
- **The Agent page, the metadata page and the tool help pages are ready to translate.**
  The Viewer's web server writes each text of the Agent page in the Viewer's language as
  it serves it (`src/web_ui/Page_Texts.py`), so the page marks its texts (`data-i18n` on
  an element whose content is one text, its titles, placeholders and screen-reader
  names, and `t()` in its scripts) and takes them from the windows' catalogs, under
  `AgentPage`, and needs none of its own; the update command collects them. What the
  Viewer sends the page comes in the same language: an agent turn's errors, model card
  problems, a refused image or Viewer capture, and a command the agent may not run. The
  terminal and MCP clients keep the English. The metadata page likewise marks its
  buttons, help, column titles, delete prompts, status and its columns' NUM and TXT
  badges, which its style now takes from the page's script, takes its texts from the
  catalogs under `MetadataPage`, and gets a refused column deletion in the Viewer's
  language. The ESMFold page stays English: it is Mol*'s own interface, which has no
  translations. A translation of a tool help page,
  `src/tools/tool_descriptions/<name>.md`, is a whole page in its language's folder,
  `src/resources/languages/<code>/help/<name>.md`, whose first line names the English
  page by its SHA-256. The Tools window shows it in its language while that line
  matches, and the English page once the English changes, until the translation is
  brought up to date; the catalog update lists those. Card titles still come from the
  English pages.
- **Benchmark.** `src/resources/benchmark/Run_Benchmark.py` times the program's heavy
  calculations on a bundled set of public sequences, so computers and versions can be
  compared. It runs ten stages, each in a process of its own with every device setting
  on Auto: sanitizing, ESM-2 8M embeddings, the all-against-all embedding alignment
  (369,370 pairs), the SSN and UMAP layouts, clustering, an embedding database search,
  injection of 92 new sequences, an embedding MSA and the BLAST all-against-all
  alignment. The report gives each stage's time, throughput, CPU time, memory peaks and
  device, every choice the tools' Auto trials made, the hardware and software, and what
  can slow a run, such as other EMAP-SSN processes or a laptop on battery. Tools report
  their Auto choices only when `SSN_BENCHMARK_RECORD` is set, as the benchmark does. A
  stage the computer can't run, such as BLAST without NCBI BLAST+, is skipped with the
  reason, and so are the stages that need it; `--stages 3,7` runs only some. The
  report, `Benchmark_Report_<date>_<time>.txt` in the benchmark folder, is written in
  the windows' language with an English `.json` of its data beside it, and is never
  deleted; the terminal shows it in English. A run keeps its files in its own `temp/`,
  which it clears, and never reads or changes `tools_settings.json` or the configured
  folders. A run took about 6 minutes on an RTX 5070 Ti with a Core Ultra 7 265KF. See
  `src/resources/benchmark/README.md`.
- **The benchmark's sequences.** 860 reviewed UniProtKB/Swiss-Prot entries of the
  metallo-β-lactamase superfamily (InterPro IPR001279, UniProt release 2026_03), one
  per distinct sequence, with 92 more held out for the injection stage. UniProt shares
  them under CC BY 4.0. The benchmark README records the query, the code that remakes
  both files byte for byte, their SHA-256 and the attribution, and
  `THIRD_PARTY_LICENSES.md` lists them.
- **A Benchmark card on the Tools window's Manual Tools tab.** **Run Benchmark** asks
  before it starts, then runs the benchmark in a console window of its own, which shows
  the progress and the report. **Report Folder** opens the folder the reports are kept
  in. The Manual Tools help page describes the stages and the report.
- MCP `emapssn_pipeline(action="start_benchmark")` queues the benchmark as a job: all
  ten stages, or the `stages` given by number or name with the stages they need. When
  the job ends, its `result` holds the benchmark's English summary: the outcome, the
  report paths, and each stage's time, throughput, device and Auto choices. `list_tools`
  describes the stages and the expected duration.
- **`run stop`.** It stops the Python command scripts the Viewer is running, with
  every process they started, and none of their commands run. With none running, it
  says so and succeeds. A script started through MCP or the Agent page reports the
  stop as cancelled.
- **`meta show {Length}`.** `meta show` and `meta display` also take the property in
  braces, as `spectrum` writes it; `meta show Length` still works. A name in braces is
  always a property, so `meta show {off}` shows a column named off instead of closing
  the display. The VR viewer accepts the same form.

### Changed

- **A fuller Help on the Meta Data page.** The dialog is wider (880 px; one column in a
  narrow window) and covers the whole page: each filter's rules beside its TXT or NUM
  badge, with whole-value wildcards, letter case, range ends, nested parentheses and
  operator precedence, cells without a number and filters that match nothing; sorting,
  resizing, moving, editing, undo and redo (the Viewer's history), deleting a column and
  copying; the Import file's layout and what Export saves; and how the table follows the
  Viewer's selection, clicks and hidden nodes.
- The Agent page is titled **EMAP-SSN Agent (Beta)**, and Clear Chat is a broom icon
  button whose name shows on hover.
- **Text on Windows is drawn without hinting.** Config, VR Config, Tools and the Viewer's
  panels draw their text unhinted on Windows, as the Script Description and the Viewer's
  web pages already did, so the bundled fonts look alike in both. Hinting had snapped
  every stroke to whole pixels, which drew Chinese noticeably heavier. Text widths change
  by a pixel here and there. macOS never hints, and on Linux the windows keep the
  system's setting, which the browser views follow too.

- **Tools' Save & Run no longer shows a message box when the tool starts.** The console
  the tool opens in shows it started. Instead the button greys out for a second after a
  click, so a double click starts the tool once.
- **The global and local alignment choices name their algorithms.** Config's and VR
  Config's Alignment Score Mode and Tools' Alignment Mode offer global (Needleman-Wunsch)
  and local (Smith-Waterman). Settings files still store `global` and `local`.

- **The Viewer's mouse and keyboard controls open from a button.** The line of key and
  mouse actions across the top of the Viewer is gone: in English it was wider than the
  default 1200 × 800 window, so its end ran under the sidebar's toggle and past the
  window's edge. An info button at the start of the threshold slider opens them on a
  card in a light tint of the theme colour, each action's keys as key caps beside what
  it does. A click outside the card, Esc, its × or the button again closes it, and the
  keyboard stays with the network, so **[ENTER]** still opens the command line. The
  command line moves up to the top of the window, level with the sidebar's toggle, and
  its box takes the theme colour: a light tint of it inside, the colour itself round the
  edge.
- **The Viewer's sidebar slides.** Its toggle slides the sidebar open and closed instead
  of switching it at once, and a click mid-slide turns it around where it is. The
  command line, the threshold slider and the status texts move with it. The sidebar is
  a card with round corners, clear of the window's edges, and each of its buttons
  centres its label in the room after its icon.
- **The Viewer's view width has limits.** The mouse wheel, a touchpad pinch and `zoom N`
  keep the view from 0.5 units wide up to the nearest multiple of 100 that shows every
  node, hidden ones included, with the margin the Viewer opens with. That widest view
  follows the nodes as they are moved, and the window's shape. A zoom stops at a limit;
  `zoom` refuses a width outside them and names both. `print`'s `zoom N`, which sets an
  image's scale rather than the view, keeps its own range.
- **The Viewer's view width is a box to type in.** The View Width at the bottom right is
  now a spin box, level with the threshold slider. Type a width and press Enter, or use
  its arrows, Up/Down or the wheel over it, 1.25 times wider or narrower a step, and the
  view zooms about its centre as `zoom N` does, within the same limits; Esc puts the box
  back. It follows the view's width as the view changes. Hidden Nodes and a metadata
  value `meta` shows stack above it, the three evenly spaced. `zoom` stays, for
  scripts, the agent and MCP.
- **The threshold is a box to type in too.** The number before the threshold slider is
  now a spin box. Type a threshold and press Enter, or use its arrows, Up/Down or the
  wheel over it, and the slider moves to it; a drag of the slider changes the box. Each
  number of the box is one of the slider's positions. Its step is the largest power of
  ten, at most 1, that still gives 1,000 steps from the lowest edge score to the
  highest, so the slider is never coarser than before, and the box shows as many
  decimals as the step has: 0.001 for scores from 0 to 9.6, 0.1 for E-value scores up
  to 300. Its lowest number still shows every edge and its highest the best ones. Esc
  puts the box back.
- **The mouse wheel zooms smoothly.** A notch zooms as far as before, about a fifth of
  the view, but over a few frames, most of it in the first tenth of a second, instead of
  in one jump. Quick notches add up, and a notch the other way takes one back. The point
  under the cursor stays put and the width limits still hold; a pan, `zoom`, the View
  Width box or a print ends the zoom where it is. Low resource mode zooms at once, as
  before.
- **Low resource mode skips the updates in between.** While the threshold slider is
  dragged its number follows, and the edges are redrawn once, when it is let go; a click
  on its track or a key still redraws them at once. The sidebar opens and closes at once,
  without sliding. A right-button drag of nodes tells the Meta Data sheet its selection
  once, on release, as a box selection already did. Without low resource mode the
  slider's edges follow the drag and the sidebar slides, as before.
- **Dragging nodes no longer resends the selection to the Meta Data sheet.** A drag moves
  the selection without changing it, so the sheet hears of it once, when the drag
  starts, instead of refiltering its rows at every mouse move and every turn of the wheel
  that rotates the dragged nodes.
- **The mouse works in the Viewer while a command is typed.** With the command line
  open, a click does what it does with the line closed: a right-click selects, drags
  and draws a selection box, a left-click focuses a node, Shift- and Ctrl-click copy a
  node's header and sequence, and hovering highlights. The command stays on the line,
  with what the click did after it until the next key, and **[ENTER]** runs it as
  typed. So a command's nodes can be selected after it is typed, and a header copied
  with Shift-click can be pasted into it with Ctrl+V.
- **The Viewer's hidden-node count takes the theme colour.** While nodes are hidden, the
  Hidden Nodes count at the bottom right shows in a dark shade of the theme colour
  instead of red, dark enough to read on the white canvas (7:1 contrast or more; dark
  grey in Black). At 0 it keeps the Text Color setting. The Text Color tip of Config
  and VR Config says so, and no longer mentions the line of instructions the Viewer
  dropped.
- **`print` finishes its file in the background job queue.** The picture is still taken
  when the command runs: the view, every tile of `print full`, or the nodes and edges
  of an SVG. Trimming and encoding the PNG, or writing the SVG, now run in the queue
  `label` and `logo` use, so the Viewer can be used meanwhile and later changes don't
  alter the file; the queue reports the saved file and opens its folder. Drawing needs
  the Viewer's OpenGL context, so keyboard and mouse input still waits while `full`
  draws its tiles. A name a queued print is still to write is refused, and timestamped
  names skip it. A PNG goes into a partial file that replaces the target when complete,
  as an SVG already did. `print full` keeps its tiles' 8-bit renders and finishes them
  in place, so it needs about 4 bytes of memory per pixel, 7 with `transparent`, where
  it took about 9; the files are byte for byte the same. A 992-megapixel `print full`
  returned to the Viewer after 6.4 s, and the job saved it 84 s later, with the process
  at 4.1 GB.
- The Network Statistics Report of Config and VR Config draws its threshold table as a
  histogram of the edges' scores, on its side: behind each threshold, a bar in the theme
  colour's tint for the edges from that threshold up to the next. Count and Percentage
  keep to the width of their figures in a wide report, so the threshold column and its
  bars take the rest, and the report's text can still be selected and copied. The report
  fits its panel at the launch size: its figures used to be spaced-out text that wrapped
  mid-row, the summary lines above the table now wrap, the report's own "Network
  Statistics" banner line is gone since the panel's title already says it, and the hint
  it shows before a report is computed wraps to the panel's width instead of being cut
  off after its first line (Qt shows only the first line of a text box's placeholder).
- **Sanitize Sequences' length histogram has a new look.** It takes the style of the
  Network Statistics Report: rounded bars in a light shade of the Tools window's theme
  colour, edged in the colour itself, on a light grid, in Noto Sans. A subtitle gives the
  number of sequences and their minimum, median, mean, standard deviation and maximum
  length, and the console prints the same line, in a run from a settings file too. With
  the length filter on, dashed lines mark its bounds. Lengths spanning 50 values or fewer
  get one bar per length, and the table a row per length; wider spans keep 50 equal bins.
  The window is titled Sequence Length Distribution. Its texts, the window's title
  included, show in the Tools window's language, Simplified Chinese in the bundled Noto
  Sans SC; the console's line stays English. It is still matplotlib; nothing new to
  install.
- **Embedding MSA's isotonic regression plot has a new look.** It takes the length
  histogram's style. The observed pairs show as hexagons shaded in the Tools window's
  theme colour by how many pairs fall in each (a colour bar gives the counts), in place
  of 5,000 blue dots; they are the pairs the fit was made on, every pair or a random
  sample of 100,000, and the legend says which. The fit is a near-black line through its
  own breakpoints, the function the missing scores are taken from, in place of a red
  line sampled at 1,000 points. A subtitle gives Spearman's ρ, R² and the number of
  observed pairs. Its texts, the window's title included, show in the Tools window's
  language.
- **The tools' pop-up charts open in the Tools window's style.** The windows of the
  length histogram and the regression plot take the Tools window's theme colour, palette
  and Noto Sans, and its icon in the title bar. Their toolbar is a light strip with
  Lucide icons, and the button of the active pan or zoom mode shows a tint of the theme
  colour. The toolbar's tips and matplotlib's own dialogs stay English. On Linux they
  take the Tools window's name, `emapssn_tools`, as their window class, which its
  desktop entry claims, so GNOME's Alt+Tab shows the Tools icon instead of a generic one
  (matplotlib names its application `matplotlib`, which no desktop entry claims).
- The Inputs & Outputs tab of Config and VR Config keeps its rows side by side in
  narrower windows: they need about 772 px (778 on Linux) instead of 1,046 px, so they
  stay side by side down to the smallest window when the statistics report is collapsed,
  and down to about 1,110 px wide with the report open. When the room runs short, the
  first column (Alignment Score Mode, Alignment Reference ID, Similarity Threshold)
  narrows before the others, and the number fields keep room for their usual values.
- The Parameter tables of Tools' Script Description give the parameter column one width
  in every table, on every page and in every language, room for any setting's name and
  key in three lines. Each table used to size the column to its own text, so names took
  up to four lines in English and six in Chinese. In a help panel too narrow for both
  columns, as Tools opens, each row stacks instead: the name over what it does.
- Faster commands on large networks. After every command the Viewer rebuilt the metadata
  table for the web page even when no page was open: 2.7 s per command at 500,000 nodes
  with 4 columns, 10.3 s with 30, paid on every line of a `run` script. It now skips that
  when no page is open and is 8–11× faster when one is. Each undo step also copied the
  whole state, one group set per node included: about 420 ms and 261 MB per command at
  500,000 nodes, up to 13 GB for the 50-step history. Unchanged arrays are now shared
  between steps and groups stored compactly: about 50 ms and 3 MB per command, and undo
  and redo take about 185 ms instead of 620–910 ms. What the page shows and what undo
  restores are unchanged.
- More Viewer commands are faster or leaner on large networks, with the same results
  (files byte for byte, labels, masks and numbering exactly):
  - `reference` reuses the alignment already loaded while the MSA file is unchanged and
    only renumbers its columns: 34 s became 0.25 s for 50,000 sequences × 3,000 columns.
    The terminal says the loaded alignment was reused.
  - `print svg` writes the file as it goes, into a partial file that replaces the target
    when complete: 14.8 s and 3.3 GB became 4.9 s and 0.15 GB at 500,000 nodes and
    5 million edges. `print full` keeps the stitched picture as 8-bit pixels: 2.4 GB
    became 0.5 GB at 58 megapixels.
  - `meta` upload reads each column once instead of each cell: 31 s became 1.2 s for
    100,000 rows × 30 properties. `meta download` builds its table a column at a time:
    10.2 s became 7.7 s at 500,000 × 30.
  - `group` copies the current groups only when a later pair reads them, and no longer
    redraws the network, which does not show groups: 189 ms became 32 ms for 150,000 of
    500,000 nodes, before the redraw it saves.
  - Expressions: header text without wildcards no longer goes through fnmatch, and `*`,
    also `query`'s default with nothing selected, matches every header at once (460 ms
    became 100 ms, and 500 ms almost nothing, at 500,000 headers). Residue groups and
    gaps such as `(RHKDE)71` and `_71` read their column once: 167 ms became 33 ms at
    50,000 sequences.
  - `cluster`/`subcluster` jaccard counts shared neighbours from the shorter neighbour
    list of each edge: the filter takes 0.30 s instead of 0.80 s at 200,000 nodes and
    7.2 million edges.
  - The first `print` of a session no longer loads pyplot, which it never used.
- **Viewer command behaviour, as decided after the command audit.** Results change on purpose:
  - **`cluster`/`subcluster` jaccard:** compares closed neighbourhoods (each node counts as its own neighbour), so an edge scores (c + 2) / (a + b − c). Isolated pairs, stars and chains are no longer forced into Noise at any threshold above 0, and a clique scores 1. Every Jaccard result changes, and the default threshold of 0.2 now keeps more edges.
  - **`cluster`/`subcluster` mcl:** each node's self-loop weighs as much as its strongest edge instead of a fixed 1, so the clusters no longer depend on the scale of the scores (identity 0–1 or −log10 E). MCL results change.
  - **`group`:** skips hidden nodes, as `color`, `spectrum` and `select` do.
  - **`logo` bits:** the small-sample correction counts the sequences with a residue at the column, as WebLogo does, with or without `id`; it counted gaps too without `id`. Bits logos of gapped columns change.
  - **`print`:** `print full zoom N` and `print svg zoom N` draw at the scale `zoom N` would set, without moving the view; a `zoom N` full print under 1000 px warns. PNGs leave out the hover colour and click rings, and keep selection borders.
  - **`run`:** a Python script runs from its own folder, and a typed one runs in the background, so the Viewer stays usable. Typed runs stop at the first failed or cancelled command and refuse scripts of more than 1000 commands, as MCP runs already did; so do VR runs.
  - **Metadata import:** in text columns, NA, N/A, None, null and NaN stay as written instead of becoming blank; number columns still read them as missing.
- **A slow Viewer launch is handed back instead of stopped.** When `start_session`'s
  wait ends while the Viewer is still loading, it returns `status: "starting"` with the
  `launch_id`, the Viewer's last output line, its process-tree CPU time and the log
  paths; the Viewer keeps loading. Cancelling the call, including a client-side tool
  timeout, also leaves it running. The Viewer now logs when it validates its inputs
  and when it builds the network display.
- `get_command_catalog` works without a running Viewer; it then reads the installed
  command sources.
- **Setup no longer repeats the readiness check.** When `.venv` needs installing or
  repairing, the EMAP-SSN and EMAP-SSN Tools launchers started from a desktop shortcut
  checked it three more times after the shortcut's own check (twice on Linux and
  macOS); they now check once more, quietly, after taking the setup lock, in case
  another launcher finished the setup first. Started from a terminal, they print why
  setup is needed once instead of three times (twice on Linux and macOS).
- **Setup stops on an unsupported system and says why.** On Windows older than
  Windows 10 version 1809 or not x64, an Intel Mac or macOS older than 14, or Linux
  other than x86_64 or with glibc older than 2.34, setup used to fail partway through
  with a package resolution error from uv. It now stops before it installs any Python
  packages and names the requirement the system misses. Setting
  `SSN_ALLOW_UNSUPPORTED_HOST=1` skips the check, for experiments on other systems.
- **Every listed Python requirement is pinned.** The ESM runtime dependencies in
  `src/esm_runtime_requirements.txt` and Markdown in `src/requirements.txt` had no
  fixed versions, so a new install took whatever was newest that day. They are pinned
  to the versions this release was tested with, like every other package the two
  files name; their own dependencies still resolve to current compatible versions. The
  first launch after updating installs the new pins and reuses the installed PyTorch
  build once it validates.
- Faster Viewer startup on large networks: edges are drawn without a per-edge Python
  loop, and the top-percent cutoff uses a partial sort. On a 12,927-node network with
  4.18 million displayed edges, headless readiness fell from 14.0 s to 9.6 s on the test
  machine. Layout generation shares the faster network preparation; cutoffs are
  unchanged.
- **Smooth threshold slider and node dragging in the Viewer.** The edges are sorted by
  score once when the Viewer opens, so the edges at or above a threshold are always the
  first ones in that order: moving the slider only changes how many are drawn, and
  nothing is filtered or sent to the graphics card again. The edge buffer holds node
  numbers, and the graphics card looks each node's position up in a small texture, so
  dragging nodes sends only the node positions (16 bytes a node) instead of every drawn
  edge. On a synthetic network shaped like an SSN, a slider step took 52 ms instead of
  2.2 s at 44,000 nodes and 48.7 million edges (9 ms instead of 203 ms at 13,000 nodes
  and 5 million edges), and a drag step 54 ms instead of 1.16 s (10 ms instead of
  104 ms), about as long as a plain redraw. Hiding or showing nodes rebuilds the edge
  buffer once. The picture is the same pixel for pixel. Sorting adds 2–3 s to opening a
  48.7-million-edge network (0.25 s at 5 million); the edges take half the graphics
  memory and about half the RAM they did. `SSN_EDGE_POSITION_TEXTURE=0` draws edges
  the old way, for a graphics driver that cannot read textures in a vertex shader.
- **Box selection no longer slows down as it grows.** A right-button box selection draws
  a border around the nodes connected to the selection, and each mouse move used to look
  up the neighbours of every selected node again, so a move cost more the more nodes the
  box already held. The Viewer now keeps, for each node, a count of the selected nodes it
  touches, and a move updates only the counts of the nodes it adds or removes. On a
  synthetic SSN-like network of 44,000 nodes and 25 million edges, a move took 11–16 ms
  however many nodes were selected, instead of 13 ms rising to 430 ms as the box grew to
  cover the network. The selection, the borders and the draw order are unchanged.
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
- **Header lists stay in their folder too.** An `@file@` name in a selection expression
  must be a plain file name in the header-list folder (`HEADER_LIST_DIR`), whether the
  command comes from the console, the web agent, an MCP client or the VR viewer. A name
  with a directory, `..`, a drive (`C:`), a rooted path (`\x`), a network share
  (`\\host\share`) or a `:` stream suffix is refused before any file is checked or
  opened, and the command changes nothing. Such names used to read files anywhere, and
  on Windows even checking whether a network-share file exists offers the user's
  credentials to that host.
- **CPU layout generation uses several cores.** The SSN physics kernel computes
  repulsion in parallel on all but two logical CPUs, as the `logo` command already
  did (set `NUMBA_NUM_THREADS` to choose the count). It also visits only pairs within
  a component and no longer allocates temporaries for every node on every step. On
  the test machine (20 logical CPUs, so 18 threads), a step took 15 ms instead of
  105 ms on a 10,000-node component, and 0.074 ms instead of 1.8 ms on a batch of 100
  small components. Coordinates are bit-identical to the serial kernel on any thread
  count. UMAP layouts with a `LAYOUT_SEED` still run on one core, because umap-learn
  makes seeded runs serial to keep them reproducible.
- **GPU layout generation is about 50 times faster and fits large components.** On
  NVIDIA and AMD GPUs the SSN physics step computes repulsion and springs with two
  kernels that EMAP-SSN compiles the first time a layout uses the GPU, with the compiler
  that ships with PyTorch (NVRTC on CUDA builds, hiprtc on ROCm builds), so nothing
  extra is installed. Repulsion no longer builds a table over every pair of nodes, so
  memory grows with nodes plus springs instead of nodes squared. On the test machine
  (RTX 4070), a step on the 12,289-node main component of a 13K network took 1.6 ms
  instead of 85 ms and 0.45 instead of 6.4 GiB, and the whole layout took 13.5 s on the
  GPU instead of 274 s on the CPU. A 44,127-node component, which needed about 88 GB of
  pair tables and failed (on Windows only after spilling 36 GiB into system memory),
  takes 6 ms per step in 1.4 GiB. Compiled kernels are kept per user
  (`%LOCALAPPDATA%\EMAP-SSN\gpu_kernels`, `~/.cache/emap-ssn/gpu_kernels`);
  `SSN_LAYOUT_GPU_KERNEL_CACHE` names another folder, and `0` turns the cache off.
- **GPU layouts change once, then reproduce exactly.** The kernels add every force in a
  fixed order instead of the previous atomic `index_add_`, so a seeded GPU layout
  reproduces run to run on the same GPU model, and differs slightly from GPU layouts
  made before. CPU layouts are unchanged.
- **The kernels check themselves before first use.** Before a GPU's first layout in each
  process, both kernels are compared with a double-precision evaluation of the same
  physics on two small test problems. These cover several tiles and a partial last one,
  split work, components, inactive nodes, the taper zone, both force caps, and tiles
  skipped beyond the cutoff. If the kernels fail to compile, load or agree, layouts use
  the previous PyTorch formulas, now in blocks of at most 16.8 million pairs so they no
  longer run out of memory, and the log says why. **The ROCm kernels have not been
  tested on AMD hardware**; this check guards them. `SSN_LAYOUT_GPU_KERNELS=0` forces
  the fallback on any GPU, and Intel and Apple GPUs always use it. The layout log names
  the implementation in use
  (`GPU layout physics on …: runtime-compiled kernels (NVRTC 13.2, sm_89)`).
- **Auto device selection ranks a whole stage.** The benchmark used to time one step
  with the setup included, so a GPU's one-time copy of the springs weighed as much as a
  step. Devices are now ranked by setup time plus a quarter of `MAX_STEPS` steady steps,
  the shortest stage the plateau test allows. A step is timed over 3 steps for
  components above 2,000 nodes and over 10 for medium batches, and the table shows
  setup, step and stage time. On the test machine, Auto picks the GPU for the main
  components of the 13K and Fungal networks and for medium batches, and keeps batches of
  small components on the CPU (0.15 vs 0.50 ms per step). The Fungal layout took 3.5 s
  instead of 29.2 s on the CPU.
- **Auto solves each spectral start once.** To benchmark the devices, Auto prepares one
  job of each size class, spectral starting layout included, and that job then prepared
  it again. The job now takes over the benchmarked batch with its own random stream, so
  seeded layouts are unchanged and the benchmark only reads the batch. The 44K network
  solves the spectral layout of its 34,302-node component once instead of twice, with
  0.4 GiB less peak memory.
- **Spectral starts use the intended eigenvectors, and large components solve them in
  seconds.** A component's starting layout comes from the eigenvectors of its normalized
  graph Laplacian after the trivial one: λ2 and λ3, and λ2–λ4 in 3D. ARPACK's
  smallest-magnitude stopping test is relative to the eigenvalue, so it often missed the
  eigenvalue of exactly zero and returned the next ones instead; skipping its first
  column then laid those components out on λ3/λ4 (λ3–λ5 in 3D). In the test networks
  that happened to three of the four components above 2,000 nodes and to about half of
  those with 20–499 nodes. The trivial eigenvector, √degree, is now excluded explicitly.
  Components of 2,000 or more nodes are solved with SciPy's LOBPCG, kept orthogonal to
  √degree, with its matrix products on all but two logical CPUs; a result that misses
  the residual or orthogonality check falls back to ARPACK, continuing the same seeded
  random stream.
- **Spectral starts no longer collapse to a dot.** Each axis of a spectral start is
  spread with asinh around its median, in units of half its 5–95% range, before the
  usual mapping into the component's box. A few outlying nodes used to stretch that
  mapping so far that the rest of a large component started inside a dot one or two
  units wide, and its explosive expansion left about a dozen nodes per 44K run on the
  layout boundary in the first stages. With both changes, the 44K network's layout shows
  no boundary warnings, and the Fungal network's time is unchanged (3.6–3.8 s). **Seeded
  layouts change once** for every component with a spectral start (4 or more nodes in
  2D, 5 or more in 3D) and for every batch that contains one, because a batch shares one
  simulation; packing may also move unchanged components. Grid starts (2–3 nodes, up to
  4 in 3D) are unchanged, and so are batches made only of them.
- **Large components get their spectral start several times faster.** For components of
  2,000 or more nodes, LOBPCG is now preconditioned with an algebraic multigrid V-cycle,
  with no new dependency. Nodes are grouped by the greedy aggregation that pyamg's
  smoothed-aggregation solver uses, ported to Numba; it gives identical aggregates.
  √degree is the near-null space. Each level is smoothed with l1-Jacobi, and a coarsest
  level of at most 500 rows is solved densely with √degree's direction removed. The
  finest-level products run on all but two logical CPUs, and every product works row by
  row, so the result does not depend on the thread count. The hierarchy draws no random
  numbers and is freed as soon as the solve returns. On the test machine the 44K
  network's 34,302-node component now solves its spectral start in about 8 s, where
  v0.3.0's solver took about 7 minutes, and the network's whole 2D layout took under
  260 s, with a 14.2 GiB peak, where it took 919 s before this release's spectral-start
  changes. On synthetic SSN-like components of 99,000 and 199,000 nodes, the solve took
  25 s and 36 s. With verbose output, the layout log names the solver used for each
  component of 2,000+ nodes, and the reason for any fallback.
- **Seeded layouts of large components come from the preconditioned solve.** Every
  component of 2,000 or more nodes, and every batch that contains one, starts from the
  AMG-preconditioned solver's eigenvectors. These match plain LOBPCG's to its tolerance
  but not bit for bit, and they are the same on every installation. With
  `SSN_SPECTRAL_AMG=0` (or `off`, `false`, `no`), or when the AMG-preconditioned solve
  fails or misses the residual or orthogonality check, LOBPCG runs unpreconditioned from
  the same start block, and the layout is bit for bit the one an unpreconditioned solve
  gives.
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
  defaults to 10 instead of 20, also in the VR Config. The grid size now only sets how
  finely components are fitted together: drawings of different components always stay
  at least half of `PACKING_PADDING` apart. A 12,927-node BLAST network that packed
  440 units across at grid size 20 packs 240 across at the new default and 165 at 2.5.
  Packed positions differ from earlier versions; saved caches keep theirs.
- **3D layouts are packed the way 2D ones are, and far more tightly.** 3D physics
  layouts used to place disconnected components on concentric spherical shells around
  the largest one, each sized as a sphere around its farthest node, so long or
  branched components left most of the volume empty. They are now fitted into a grid
  of `PACKING_GRID_SIZE` voxels by the 2D packer's rules, with the same guaranteed
  clearance of half of `PACKING_PADDING`. A component also claims every gap in it up
  to about twice `PACKING_PADDING` wide and every space it encloses, so no component
  lands inside another one's cage or in its narrow gaps. `PACKING_GEOMETRY` now applies
  in 3D, a cube for `Square` and a ball for `Circle`, and the VR Config shows its
  Packing Geometry dropdown again. With the default settings, a 4,070-node network
  packs 93 units across instead of 275 and a 12,927-node one 83 instead of 202, in
  under half a second. Packed 3D positions differ from earlier versions; saved caches
  keep theirs.
- **Tiled embedding alignment is about 1.8 times faster.** `Align_Similarity_Matrix` and
  `Network_Injection` built every padded GPU batch with one copy per target and sent the
  target lengths from unpinned memory, which made the CPU wait for the GPU on every
  batch and left the GPU half idle. Each tile's embeddings are now uploaded once into a
  packed GPU buffer that every batch gathers from in one step, lengths travel through
  pinned memory, one CPU task aligns a whole batch instead of one task per pair, and
  tiles no longer wait for the GPU queue to empty when they change. On the test machine
  (RTX 4070, i5-13600K), an MCP job on 12,927 sequences took 31.4 s per 500,000-pair
  batch instead of 56.9 s. The old code's speed also varied with CPU load (up to 88 s
  per batch in another session); the new code is limited by the GPU. Scores and
  alignment lengths are unchanged bit for bit for a given execution plan. The
  `GPU_STREAMS` setting of `Align_Similarity_Matrix` and the `ACCELERATOR_LANES` setting
  of `Network_Injection` are gone.
- **Faster alignment scoring when the CPU sets the pace.** A float32
  dynamic-programming kernel aligns pairs about 2.2 times faster per core when float32
  is provably exact: a global gap penalty of 0 with a local gap penalty of 0 or a
  negative power of two, which includes the defaults (0 and -2). Other gap penalties
  use the previous kernel. With 2 CPU workers, the first batch ran at 7,255 instead of
  3,326 pairs per second, with identical scores. This matters with GPUs much faster
  than the test machine's, where the CPU alignments limit throughput.
- **Labels, switches and buttons size themselves to their text,** so longer wording,
  such as a translation, moves or widens them instead of being cut off. In the Config
  and VR Config windows, fields start one spacing after the longest label instead of
  at a fixed 180 px column, and every tab still shares that column. The ON/OFF and
  Auto switches, the color **Pick** buttons and the Viewer's sidebar buttons keep
  14 px between their text and each end, and a switch keeps one size in both states.
  In English the label column is 174 px, the ON/OFF switches 51 px and the Viewer's
  sidebar 137 px.
- **Tools window cards have a left and a right section.** The left section holds a
  card's **Save & Run** and **Export** buttons and its field labels, and is as wide as
  the widest of these across all tabs; the title and the fields start one spacing
  after it. The buttons, and **Save Directories**, fit their text with 14 px at each
  end. **Export Setting** is now **Export**, in the same font as **Save & Run**, and
  its tooltip says the JSON file goes to the Setting Export Directory and that the
  tool does not run.
- `label` no longer opens the network file when it queues a report. It refused to run
  when that file had been moved, deleted or replaced after the Viewer loaded it,
  although the analysis uses only the loaded data. The workbook still names the
  configured network file.
- In Viewer selection expressions, numeric `!=` no longer selects nodes that have no
  value, like the other comparisons and ranges. `{GRAVY!=0}` used to include every node
  without a GRAVY (a sequence with no scored residue, or an empty cell in an imported
  column). `!{GRAVY=0}` still selects all other nodes, those included. Text `!=` is
  unchanged: it selects every node the containment test rejects, blank values included.
- `subcluster` numbers subclusters by size, as `cluster` numbers clusters:
  `subcluster_N_1` is the largest, and equal sizes are ordered by their lowest member
  node. They used to be numbered in the order they were found.
- On Windows 11 25H2 or newer, the AMD Radeon RX 7600 XT and RX 7650 GRE, the laptop RX
  7600S, RX 7600M, RX 7600M XT and RX 7700S, and the Radeon PRO W7600 and W7500 now get
  the ROCm backend, like the RX 7600, whose gfx1102 chip they share. AMD's ROCm 7.14
  support list names only the RX 7600, so the others ran on the CPU or, beside some AMD
  processors, took the target of the processor's graphics (see Fixed). The 7700S was
  taken for an RX 7700, a gfx1101 GPU, so the wrong ROCm build was installed, failed
  validation and left it on the CPU. On such a PC, the first launch after updating
  installs ROCm and keeps it if it passes validation; other PCs are unaffected. On
  Linux, the RX 7600M and PRO W7500 were recognized only when the ROCm runtime reported
  their target; their `lspci` names now map too, so a new installation without the
  runtime also tries ROCm. For the other cards on Linux, see the Ubuntu 24.04 entry
  under Fixed.
- On Linux, an AMD GPU's GFX target now comes from the kernel. amdgpu's ROCm driver
  (KFD) lists each GPU it can run ROCm on, with its target and PCI address, in
  `/sys/class/kfd`, so no ROCm software has to be installed. Before, without `rocminfo`
  or `rocm_agent_enumerator`, a GPU got ROCm only if GPU detection knew its `lspci`
  name, and then only provisionally; Ubuntu 22.04's older PCI ID list names no recent
  card at all. A computer with GPUs of several targets got none of them from the ROCm
  runtime, which does not say which GPU has which. Each GPU now gets its own target,
  whatever its name suggests, and a GPU the kernel lists no ROCm node for gets none.
  Where the kernel reports no targets, the ROCm tools and the name decide as before.
  After updating, an installed ROCm backend is validated and kept without reinstalling.
  A computer that used the CPU because its GPU was unnamed, or only provisionally
  recognized, installs ROCm on its first launch and keeps it if it passes validation.
- Linux ROCm is no longer for Ubuntu alone. Distributions built on Ubuntu 22.04, 24.04,
  25.10 or 26.04, such as Linux Mint and Pop!_OS, get it by the Ubuntu release they name
  (`UBUNTU_CODENAME`). The Instinct MI300 (gfx942) and MI350 (gfx950) series also get it
  on the other releases AMD's ROCm 7.14.1 matrix lists for them: RHEL 8.10, 9.4, 9.6,
  9.8, 10.0 and 10.2; Oracle Linux 9 and 10 (and 8 for the MI300); Rocky Linux 9 (MI300
  only); Debian 12 and 13; and SLES 15.7 and 16.0. These releases are listed upstream
  but not tested by the project; the tensor validation after installing decides, as
  everywhere.
- Intel GPUs are now recognized by PCI device ID, and are eligible for the XPU backend
  when the pinned PyTorch 2.12.0 XPU build carries compiled kernels for them on that
  operating system. Before, a GPU needed "Arc" in its name, which depends on branding:
  Intel's driver calls a Meteor Lake-H GPU "Intel Graphics" when its memory is
  single-channel, and the PCI ID list of Ubuntu 24.04 calls Lunar Lake and some Arrow
  Lake-H and Meteor Lake-H GPUs "Intel Graphics", one Arc A-series GPU "DG2 [Arc Xe
  Graphics]", and knows no Battlemage GPU at all. These now use the XPU backend, as do Meteor Lake-U
  GPUs on Windows, where the wheel carries their kernels too. Two kinds of GPU the old
  names admitted are now left out because the wheels have no kernels for them: a Data
  Center GPU Max on Windows, and the Max 1550VG. On Windows, an Intel GPU that runs
  without Intel's driver (Microsoft Basic Display Adapter) is not eligible, as the XPU
  build would fail its validation. After updating, an installed XPU backend on a GPU
  that stays eligible is validated and kept without reinstalling.
- Commands in an Agent reply no longer all run automatically. Commands that only change
  the view or the session's in-memory state still do. `save`, `select save`, `print`,
  `export`, `logo`, `label` (except `label reset`), `esmfold`, `run`, `alignment`, and
  `meta` uploads, downloads and file imports wait in the Agent page until you press
  **Run** or **Discard**; discarding one skips the rest of that reply. While one waits,
  later commands, including MCP requests, queue behind it, as they do behind a file
  dialog. A reply's `agent` lines are refused. Commands typed in the Viewer or sent
  through MCP are unaffected, and `get_command_catalog` reports each command's
  `agent_policy`.
- The Agent page switches only to a saved model card, named by its id, so a request can
  no longer point the agent at an unsaved server. Edits to a card take effect once
  saved, and the page says so when the model being switched on has unsaved edits.
- The Configuration, Tools and VR Configuration windows, the Viewer's sidebar and
  slider, and its Agent and Meta Data pages share a new look: light greys, hairline
  borders, rounded controls on rounded cards, tabs as a segmented control, dropdowns
  that open their list below the field with up to ten entries in view, switches for the
  ON/OFF toggles, and one main button per window (such as **Save & Run**), in the
  window's theme colour with its focus rings and sliders (see Theme colours). Other
  buttons, **Exit** included, are outlined. The Agent page keeps its dark mode. On the
  Fold View page Mol* draws its own interface; only the page's background, fonts and
  status toast take the new look. Line icons from the open-source
  [Lucide](https://lucide.dev) set (ISC licence, bundled in `src/resources/icons/`)
  replace the emoji the windows used as icons, so they look the same on every system:
  the folder buttons, the language globe, the Tools card titles and help headings, the
  Viewer's sidebar buttons, and the web pages' controls. The tool help pages
  (`src/tools/tool_descriptions/*.md`) no longer contain emoji; the Tools help pane
  draws the icons. Sliders no longer show tick marks. The tool scripts' printed ✅, ❌ and
  ⚠️ markers are unchanged.
- Tools' **Exit** button fills the row under the Script Description, up to the
  Language dropdown, so the tips card reaches the bottom of the window.
- The **Pick** buttons of Config and VR Config open a colour picker on a card under the
  button, in place of Qt's colour dialog: a square of saturation and brightness in the
  colour's hue, a hue strip, the colour's hex code to type, and preset colours with the
  Viewer's defaults among them. Each choice fills the field at once; Enter or a click
  outside closes the card, and Esc puts the field's colour back.
- **Leiden results depend only on the network.** `cluster` and `subcluster` sort the
  edges into one order, each as its lower and higher node, before Leiden runs. Its
  seeded partition used to change with the order and direction a cache stores the
  edges in (130 against 129 clusters on one reordered network), so Leiden results
  change once. On 48.7 million edges the sort takes 0.2 s when they are stored in node
  order and up to 10 s when they are in random order.
- **MCL starts from single precision in `subcluster` too.** `cluster mcl` built its
  matrix from float32 scores and `subcluster mcl` from float64 ones; both use float32
  now. MCL works in float64 from its first pruning on, so only its first round's
  rounding changes, and a subcluster changes only where that tipped a node.
- **`print full` refuses an image of more than 1000 megapixels.** It works out the
  image's size before drawing any tile, and refuses a larger one, with or without
  `zoom N`, naming the smallest N that fits. It used to start anyway, and could run
  for hours or fail only when memory ran out. The limit is `MAX_FULL_IMAGE_MEGAPIXELS`
  in `src/commands/print.py`; making the image takes about 4 bytes of memory per pixel,
  7 with `transparent`.
- The Fold View no longer contacts a public server each time it opens. Mol* listed the
  public saved states at `webchem.ncbr.muni.cz` on every load, so opening the page sent
  a request there even when nothing was downloaded. That list is turned off, so the
  Fold View's state panel no longer shows remote states. Downloading a structure, map
  or saved state on request still contacts the server it comes from.
- Third-party notices: the KaTeX fonts are listed under the SIL Open Font License 1.1,
  which their files state, with its text in `src/resources/fonts/LICENSE.OFL`; they had
  been listed as MIT only. `src/resources/esmfold/NOTICES.molstar-bundle` gives the
  copyright notices of the 78 packages compiled into Mol*'s bundle, among them RxJS
  (Apache-2.0). `THIRD_PARTY_LICENSES.md` credits pyamg (MIT) for the ported
  aggregation of the layout's spectral start, the Noto licence files carry the fonts'
  own copyright lines, and the AMD ROCm packages are listed for the Ubuntu releases
  that get them.
- **`undo N` and `redo N` report once.** They print only "Undid N steps." or "Redid N
  steps.", on the console line too. The steps are quiet, and the Viewer redraws the
  metadata HUD and updates the web pages once at the end (the VR viewer sends the
  headset the result once). A plain `undo` or `redo` reports as before.
- **Each copy of EMAP-SSN opens its own windows.** Configuration, VR Configuration and
  Tools allowed one window each per user, so launching one from a second checkout,
  such as a fresh clone beside your usual folder, brought the first checkout's window
  forward, and that window reads and saves the first checkout's settings files. A
  launch now hands over only to a window of its own checkout, and says so in its
  terminal instead of exiting without a word: "EMAP-SSN Tools is already open from this
  checkout (<folder>); brought that window to the front instead of opening another." A
  window that was already open when you updated is not recognized, so close it before
  launching again from the same folder.

### Fixed

- A column width dragged on the Meta Data page snapped back whenever the browser window
  was resized: the spreadsheet fitted every column to its contents again at each redraw.
  Columns now keep their widths through resizes, and a dragged width also outlasts the
  Viewer's updates of the table (after a command, an undo or an import), while the other
  columns still fit their new values.
- A metadata column moved on the Meta Data page went back to the Viewer's order at the
  Viewer's next update of the table. It now keeps its place, and a column deleted and
  brought back by an undo returns to where it was; columns an import or a command adds
  come last.
- After the Meta Data page lost its connection to the Viewer and reconnected, its buttons,
  such as Import, and its Ctrl+Z and Ctrl+Y shortcuts sent their action once more for each
  reconnect. A reconnect now refreshes the table the page already shows.
- **Viewer command audit.**
  - **`label`:** it kept dropping fully conserved residues at exact thresholds: with `id` weighting, a residue in every member could fail `cmin 100%`, and 999 of 1000 failed `cmin 99.9%`. Hex Color now matches the viewer's palette, and thresholds outside 0–100% are refused. `--help` and typos such as `gmax=0.4` no longer become report filenames. With nothing to analyse, it fails before writing an empty workbook. The ambiguity codes X, B, Z and J are no longer reported as conserved residues; they still count as occupied positions.
  - **`query`:** grouped targets such as `(AC)<=30%` are exact at the boundary, and multi-word metadata values such as `{Organism=Escherichia coli}` work in `query` and `logo`.
  - **`zoom`:** `zoom N` gives a view exactly N wide, not 1.1·N.
  - **`cluster`:** a NaN or infinite Leiden resolution or weight is an error instead of a crash that escaped every handler. MCL refuses non-positive scores and keeps its minimum size when clusters overlap. Reclustering removes the previous clustering's `subcluster_N_M` groups, which went on naming nodes of clusters that no longer existed; undo restores them. `reset clusters` and `label reset` remove them too.
  - **`spectrum`:** a colormap name in another case, such as `Viridis`, works, and an unknown name is an error instead of a silent switch to coolwarm.
  - **`reference`:** a node with no sequence in the alignment is refused, instead of replacing a working reference with occupancy numbering. When several headers match, one with a sequence in the alignment is used. When the alignment cannot be reloaded, it says why.
  - **`logo`:** a mistyped keyword such as `nogap` no longer becomes the file `nogap.svg`, overwriting any file of that name: a filename must end in `.svg` or `.png`.
  - **`alignment`:** says why a file was rejected, such as unequal sequence lengths or an empty file.
  - **`esmfold`:** no longer leaves its temporary input file behind when PyTorch is missing.
  - **`run`:** treats `#` lines as comments, and a failed typed run names the command it stopped at and how many commands it did not run.
  - **Metadata:** a column named `id` no longer redirects web-table edits to the wrong node. The advertised `.xls` uploads work: `xlrd` is now a requirement.
  - Commands run for MCP or the agent page report whether their browser page opened, and never wait on a dialog.
  - **VR:** `label` and `logo` work, `hide` reaches the headset, and undo restores metadata.
  - **Smaller command fixes:** `cluster`, `subcluster`, `print`, `hide`, `select save` and `reference` refuse arguments they would have ignored or folded into a file name, naming the first; `subcluster` wants `cluster_1`, not `cluster_01`, and changes nothing when no subcluster reaches MIN_SIZE. With MIN_SIZE 1, Leiden makes isolated nodes singleton clusters, as MCL and Jaccard do. `print foo.svg` saves an SVG and says when it replaced a file. `select save` writes FASTA for `.fa`, `.faa` and `.fas` too, and `@name@` reads them. Two different `select` modes are refused. `undo N` and `redo N` step up to N times. Property names with `.` work in `{...}` expressions.
  - **Smaller output fixes:** a PNG logo too large to draw (over 370 positions) is refused before its job starts, and a long reference header no longer widens a logo. `query` sums residues under 1% on an "other" line and fails, like `logo`, when the matched nodes are not in the alignment. `esmfold` names the clicked node it folds when nothing is selected, and refuses a second worker while one runs. `spectrum` restores its displayed property on undo, and undo and redo leave a metadata HUD closed with `meta show off` closed until the next `meta show` or `spectrum`; undo no longer changes the sidebar, and commands reopen it only to add a button. The agent waits 300 s for a slow model, not 15 s. VR: `meta download` filters may hold spaces, and the terminal notes changes the headset cannot show.
- `print svg` looked different from `print` of the same view. Nodes were not drawn in
  screen order, `+`, `|` and `-` shapes lost their colour, and node sizes and edge
  widths came out about twice as thick at a typical zoom. Outlines were thinner than on
  screen, most of all on a high-DPI screen, where they were about half as thick: VisPy
  fades an outline over a device pixel on each side, so a 0.5-pixel one looks about 1
  device pixel wide (0.98), and the SVG wrote 0.5. Outlines are now as wide as they
  look, edges as wide as GL draws them (a whole number of device pixels, at least one),
  and selected nodes keep their border. The view box frames the network as the PNG is
  trimmed, with 20 of the PNG's pixels around what is drawn, instead of 5 % of the
  network's size. Every marker is drawn as VisPy draws it: `cross`, `x`, `vbar`, `hbar`
  and `ring` were thin strokes without an outline, `arrow` a triangle, `clobber` a
  pentagon, and the star, triangles and `cross_lines` had other proportions.
- MCP `start_session` terminated a healthy Viewer after a fixed 30 s. On a
  12,927-node network loaded from OneDrive, the Viewer was stopped while building its
  display, seconds from publishing its session. A readiness probe that timed out while
  the Viewer's Qt thread was busy also stopped the launch; the probe is now retried.
- MCP pipeline jobs inherited the server's standard input, which carries the MCP
  protocol; a job that prompted for input could consume protocol messages.
- On Windows, cancelling an MCP pipeline job, or closing the MCP server while a job ran,
  could leave processes the job had started running, such as the worker processes of an
  alignment with several `WORKERS`. The server ended only the process it had launched,
  which in a virtual environment is the `python.exe` redirector. The tool's interpreter
  stopped with it, but the interpreter's own child processes did not. The job's whole
  process tree is now ended. The server lists the job's processes itself, before
  stopping any, rather than asking `taskkill`, which stops nothing when Windows
  Management Instrumentation (WMI) is over its memory quota on a busy computer. A stop
  that leaves a process running is logged, and the job's `failure_message` names it.
- On Linux and macOS, cancelling an MCP pipeline job, or closing the MCP server while
  one ran, left a process of the job running if it ignored the stop signal (SIGTERM):
  the job's main process stopped, and the forced stop (SIGKILL) followed only when the
  main process itself outlived the grace period. Every process of the job now gets the
  rest of the grace period, and any still running after it is killed. A process that
  cleans up on SIGTERM still finishes first; multiprocessing's resource tracker, for
  example, removes a cancelled pool's semaphores from `/dev/shm`.
- Embedding MSA, embedding search (SSEARCH) and pairwise embedding alignment (PWA), run
  through MCP or with a hand-written settings file, accepted settings that the Tools
  window, the Viewer and layout generation refuse. Embedding MSA and SSEARCH accepted
  local scores with `alignment_length` normalization, dividing each local score by its
  own alignment length; their MCP schemas now require `shorter_sequence`,
  `longer_sequence` or `average_sequence` with local scores. An `ALIGNMENT_SCORE` (MSA)
  or `ALIGNMENT_MODE` (SSEARCH, PWA) other than `global` or `local` was read as one of
  the two: SSEARCH ran a global search, with the global gap penalty, for any value but
  `local`, such as `Local`; Embedding MSA built its guide tree from local scores for any
  value but `global`, such as `Global`, even with `alignment_length` normalization; and
  PWA ran a local alignment, with the local gap penalty, for any text but `global`, such
  as `Global`, while its report printed that text as the mode. SSEARCH divided every
  score by its alignment length when `NORM_MODE` was not one of `alignment_length`,
  `shorter_sequence`, `longer_sequence` and `average_sequence`, such as
  `Longer_Sequence`, while its reports printed that text as the mode, and Embedding MSA
  refused an unknown `NORMALIZATION_MODE` only after loading the network, and accepted
  one with a BLAST network. All three tools now stop with a configuration error that
  names the setting, before opening their inputs.
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
- On Linux, a job that Config or MCP opened in a terminal could run with another
  program's environment. Ptyxis, the default terminal of Ubuntu 26.04 GNOME, runs each
  new window's command with the environment of whichever program started the running
  Ptyxis. Config's **Save & Run** with **(New Layout Cache)** therefore started the
  layout generator without the settings Config had prepared: the Viewer opened with
  default settings, Low Resource Mode off among them, and Config's settings snapshot
  stayed in `/tmp`. While an earlier job's window was open, the next job got that job's
  settings file and layout cache instead. A normal-mode MCP `start_session` never found
  its Viewer. Every command started in a terminal on Linux, Tools' included, now runs
  with its caller's complete environment and in its caller's folder, which a short
  first step reads from a private file and then deletes; only `TERM` and the other
  variables that describe the terminal come from the terminal. The environment stays
  off the command line, where `ps` would show it to other users, and a terminal that a
  job starts no longer passes that job's settings to windows opened later.
- On Linux and macOS, every job opened in a terminal left a finished process behind (a
  zombie, such as `[ptyxis] <defunct>`) until Config or Tools exited: the terminal
  program, which a single-instance terminal ends at once, was never collected. It is
  now collected as soon as it ends.
- On Windows, the Viewer, Config, Tools and the layout generator failed when they
  printed text outside the Windows ANSI code page (cp1252 on most Western systems),
  because Python writes pipes and log files in that code page. A Viewer started by MCP
  `start_session` failed any command that printed such text, such as a selection error
  echoing an expression with `α-amylase` in it, or the "LLM Agent Activated: … → …"
  line, with "'charmap' codec can't encode character", while `read_log` and the Viewer's
  console window read UTF-8, so even `é` came back as `�`; the normal-mode console
  window also lost or garbled a character that a pipe read split in two. Config and
  Tools opened from the EMAP-SSN and EMAP-SSN Tools shortcuts, and MCP layout jobs,
  failed the same way: Tools could not run any tool from a project folder such as
  `Projekt-α`, Config closed when it found two compatible cache folders for the selected
  inputs and their paths had such a character, a `start_layout_job` job reading or
  writing such a path failed, and the terminal that shows the log after a failure read
  it as UTF-8. A layout generator run on its own
  (`python src/Layout_Cache_Generator.py <settings.json>`) with its output redirected
  ended before saving its cache when the FASTA file was in a `Projekt-α` folder. All
  four now write UTF-8 to files and pipes, whatever started them, and escape characters
  a terminal's encoding lacks instead of failing, and the console window shows a split
  character whole.
- On Windows, the `run` command mangled non-ASCII text in the commands that a Python
  command script printed, however the Viewer was started. Python wrote the script's
  output to the Viewer's pipe in the Windows ANSI code page (cp1252 on most Western
  systems), while the Viewer read it as UTF-8. A script printing `α-amylase` failed
  with "'charmap' codec can't encode character", and `é` vanished from a typed `run`
  (`select "café"` ran as `select "caf"`, which also selects "caffeine") or became
  `�` in a request from MCP or the agent page. Scripts now print UTF-8 whatever the
  Viewer's environment says, and a byte that is still not UTF-8 shows as `�` instead
  of disappearing. The VR viewer's `run` does the same.
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
- `inspect_file` reported valid 3D layout caches (for the VR viewer) as invalid.
- A network whose `model_name` attribute was not valid UTF-8 was accepted when the
  attribute was a variable-length string, h5py's default for text: h5py reads the
  invalid bytes as lone surrogate characters, which then reached cache names. Such a
  network is now refused with the "not valid UTF-8" error that a fixed-length attribute
  already got.
- `capture_view` on a headless Viewer failed with a bare OpenGL error; it now explains
  that headless Viewers cannot render on Windows and that normal mode can.
- `read_command_output` returned an empty page that never advanced when its `limit` was
  smaller than the next character. With a limit of 1 to 3 bytes, a character such as
  `é`, `名` or `🧬` gave no text, a `next_offset` equal to `offset` and `eof` false, so a
  client reading page by page looped forever. The limit is now 4 to 32,768 bytes, enough
  for any UTF-8 character. MCP clients were unaffected: their schema already required at
  least 4.
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
- An agent model card with a blank Model and a URL ending in the chat endpoint, such as
  `http://localhost:1234/v1/chat/completions`, kept the model name `default`: the Viewer
  asked for the server's model list at `…/v1/chat/completions/models`, which servers
  answer with 404, instead of `…/v1/models`. With a trailing slash
  (`…/chat/completions/`), every chat request also went to
  `…/chat/completions/chat/completions`. Both forms now behave like the `…/v1` base URL.
- The Viewer's web server no longer serves files from outside a route's folder. On
  Windows, rooted (`\Users\…`) and drive-relative (`C:..\…`) request paths reached files
  and folders beside a route folder whose names begin with its name, such as
  `Predicted_Structures_old` beside `Predicted_Structures`. On every platform, symbolic
  links and junctions inside a route folder were followed wherever they pointed. Such
  requests now get 403. A route folder that is itself a link keeps working, and a file
  whose name only starts with two dots, such as the `..x.pdb` structure of node `..x`,
  is now served.
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
- `Align_Substitution_Matrix.py` (BLAST all-vs-all) exited with code 0 when no input
  FASTA was selected, so MCP reported a job that wrote nothing as succeeded. It now
  exits with code 1.
- On Windows, `Align_Substitution_Matrix.py` (BLAST all-vs-all) stopped within seconds
  with `Argument "out". File is not accessible` when its network folder was deep, even
  without spaces. `blastp` cannot open a path of 260 characters or more, whatever the
  long-path setting, and its result files add 92 characters to the temporary
  workspace, so a workspace path of about 170 characters already failed. Such a
  workspace now reaches BLAST through its 8.3 short path or a temporary directory
  link, as a workspace path containing a space already did. When neither can be made,
  the error gives the path's length and suggests mapping a drive letter to the network
  folder.
- Embedding MSA ran out of memory on large complete networks. Filtering stored every
  edge as a Python tuple (about 130 bytes each), so a 44,127-sequence network with 973.6
  million edges needed about 127 GB and stopped with `MemoryError` on a 96 GB PC. Edges
  are now filtered as whole arrays, and arrays are freed once used. Bootstrap workers
  pass SciPy a float64 matrix directly instead of a float32 one it must convert, and
  tree distances are summed in place. That run should now peak at about 51 GB with
  `WORKERS` 3, about 16 GB per worker; guide trees are unchanged bit for bit.
- `Sparse_MSA_Converter.py` exited with code 0 when it converted nothing: when its
  input FASTA was missing, validation rejected the alignment, or the HDF5 save failed.
  An MCP job therefore reported `succeeded` (seen when step 4 failed and the queued
  converter found no alignment). It now exits with code 1. With `CONVERT_ALL`, it
  exits with code 1 when any alignment fails, after converting the rest, or when
  `MSA_DIR` holds no `.fasta` files.
- `Network_Injection.py` stopped every run with `TypeError` before aligning anything
  (v0.2.0 and v0.3.0): it checked the precision of an existing output network before
  setting that network's path. Past that check, CPU runs failed too, because the worker
  processes never received the gap penalties read from the input network. Both are
  fixed, so runs complete. Its final compile step now also stops the job with exit code
  1, naming the file, on a batch file it cannot read, instead of leaving that batch's
  pairs in the network with zero scores, and writes the network to a `.partial` file
  that replaces the final one only once complete.
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
  Occupancy % and Alignment Offset fields stack, the Alignment Offset field stayed 100
  px wide instead of reaching the right edge like the others. It now does; in a wide
  window it keeps a narrow width beside the other two.
- Config and VR Config could point the Target Cache, and so Save & Run, at the layout
  folder of inputs that were no longer selected. When other inputs were chosen while the
  selected FASTA or network was still being checked, and those inputs had been checked
  before, their folder showed at once; but the earlier check went on, and when it
  finished it replaced that folder with the one for its own inputs. Its result was also
  kept for the newly selected files, so the wrong folder came back whenever they were
  selected again, until Config was restarted. Choosing other inputs now stops the
  earlier check and ignores its result, and a file that changes while it is being
  checked is checked again at the next change in the window instead of keeping the
  result for its earlier content.
- Config and VR Config missed a layout cache published after they checked the selected
  inputs. When the Layout Generator that Save & Run started (or any other run) finished,
  the Target Cache still read [Needs Computing] with only (New Layout Cache) to choose,
  so a second Save & Run computed the layout again, until the inputs were chosen again.
  Both windows now look at the layout folder again every second while they are shown,
  and whenever the cache list opens: a new cache shows at once and is selected, a cache
  that goes leaves the list, and while nothing changes the window is left alone, keeping
  the cache chosen or the new cache name being typed. Only folder listings are read for
  this, the full search runs only when one changes, and a layout folder that is slow to
  list is looked at less often. A cache folder removed while Config listed its caches
  now shows as a cache error instead of closing Config.
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
- In Viewer selection expressions, quotes around a text value with a `*` or `?` wildcard
  became part of the pattern, so `{Organism="*coli"}` matched no node and
  `{Organism!="*coli"}` selected every node. Quotes are now ignored with a wildcard, as
  they already were without one.
- The Config and VR Config accepted a saved profile named "(custom)", "(default)" or
  "(new)". The profile selector listed it under the same text as the built-in entry
  and could not tell the two apart. These names are now reserved like "custom",
  "default" and "new", and an existing profile file with one of them no longer
  appears in the selector; rename the file to use it again.
- `label` reported each failure to MCP clients twice, once with and once without the
  `Error:` prefix, for example when an argument was invalid or no clusters or groups
  existed. Each failure is now reported once, as the console shows it.
- `cluster mcl` and `subcluster mcl` check the inflation against its documented range,
  1.1 to 10.0, before anything changes, and the error names that range. A value outside
  it used to reach the clustering library and fail with its bare "Invalid inflation
  parameter".
- The Config and VR Config could not save while a BLAST network was selected: **Save**
  reported "invalid value for ALIGNMENT_SCORE" and **Save & Run** stopped before
  launching. A BLAST network blanks the Alignment Score Mode and Normalization Mode
  fields, which BLAST scores do not use, and the blank fields were saved as empty text.
  Saving now keeps the choice the fields showed before they went blank, or that of a
  profile loaded since. Selecting an alignment network again restores that choice
  instead of resetting it to global and alignment_length.
- In Config and VR Config, resting the pointer on an input or its label for about
  0.7 s replaced whatever the tips panel showed, so the result of a **Consistency
  Check**, a network-statistics message or **Save**'s report could vanish before it
  was read. Hovering now shows tips only while the panel holds a tip. After such a
  message, a left click on an element with a tip, or moving to it with Tab or a
  shortcut key, shows that tip and lets hovering show tips again. Focus that returns
  by itself, as when the window is activated again, no longer replaces a message
  either.
- Saving settings could wipe saved data. A `tools_settings.json` with one syntax error,
  such as a trailing comma, counted as empty: **Save Directories** then kept only the
  directories, and a tool's **Save & Run** only that tool's section, deleting every
  other tool's settings while reporting success, and a failed write could leave the file
  empty. The agent panel replaced every saved model card and API key with the three
  local defaults when a save request had no `cards` list, or when it was opened while
  `model_card.json` could not be read. Both files are now refused with the parse error
  and left unchanged, saves replace them atomically, and the agent panel shows "Saved"
  only once the Viewer confirms the save.
- A save interrupted by a crash or a forced stop could leave a temporary copy beside the
  Biohub API token file, which holds an API key, or beside `viewer_settings.json` or the
  VR Config's `viewer_settings_vr.json`, and git offered the copy for commit. The atomic
  saves of `model_card.json` and `tools_settings.json` (the previous entry) use
  temporary copies too. All of these copies are now git-ignored, and the next save or
  load of the model cards or the Biohub token, and the next save of either Config
  window's settings, removes copies more than a minute old without reading them.
- The Tools window opened over an unreadable `tools_settings.json` showing default
  values without a word; the problem came to light only when **Save Directories** or
  **Save & Run** refused. It now warns when the window opens, with the parse error's
  line and column, and leaves the file unchanged.
- The Tools window's file dropdowns did not use the saved directories until they were
  opened. Each listed its default folder instead, so with custom directories it showed
  no file, or one from the default folder; once opened, it listed the saved directory
  but selected its first file rather than the saved one. **Save & Run** saved and ran
  whichever file it showed, and the Embedding MSA card lost its saved Alignment Score
  Mode and Normalization Mode. The dropdowns now open on the saved file. A relative
  directory is found from the EMAP-SSN folder, as the tools find it, rather than from
  the folder the window was started in; a blank one lists the default folder; and the
  📂 button beside a dropdown opens the folder it lists.
- A file saved for a Tools dropdown that was no longer in its folder, deleted or moved,
  was replaced without a word by the folder's first file, which **Save & Run** then
  saved and ran. It now shows as "Unavailable saved file [name]", as a removed model
  does, and stays selected when the dropdown is opened again; **Save & Run** refuses it
  until another file is chosen or the file is back. A file saved by its full path,
  which the tools find though no folder lists it, now shows as it is instead of the
  folder's first file.
- On Windows, the Tools window's Script Description could turn partly or wholly black,
  sheared into a chessboard of garbage pixels, after it widened in one jump (closing the
  settings panel, for one), and stayed so until the page drew again. Chromium now draws
  the help page in software, as it already did on Linux: in this release's tests, GPU
  drawing corrupted the page after up to one in eight such jumps, and software drawing
  after none of 320.
- A tool run from the command line without a settings file, while `tools_settings.json`
  could not be read (invalid JSON, empty, or not a JSON object), printed a warning and
  ran on its built-in defaults; an MCP job given parameters without every directory
  likewise ran on the default directories. Both now stop before opening any input,
  naming the file and the parse error's line and column, and leave the file unchanged. A
  missing `tools_settings.json` still means the defaults.
- On Ubuntu 24.04, an AMD Radeon RX 7600, RX 7600 XT, Radeon PRO W7600 or laptop RX
  7700S never got the ROCm backend. Ubuntu 24.04's PCI ID list gives all of them the
  name "Radeon RX 7700S/7600/7600S/7600M XT/PRO W7600", which GPU detection read as an
  RX 7700, a gfx1101 GPU. If the ROCm runtime was installed, ROCm was refused because
  the runtime reported gfx1102; if not, the gfx1101 build was installed and failed
  validation. Either way the card ran on the CPU. That name now maps to gfx1102. Where
  `rocminfo` or `rocm_agent_enumerator` reports gfx1102, the first launch after updating
  installs ROCm and keeps it if it passes validation. Without the ROCm runtime, a new
  installation tries ROCm, but an existing one stays on the CPU until you run
  `python src/Install_Dependencies.py --refresh-backend` in its environment.
- On Linux, GPU detection never found an AMD Instinct MI300A, MI300X, MI308X, MI325X,
  MI350X or MI355X, so a node full of them used the CPU. `lspci` lists these as
  "Processing accelerators" rather than as a display controller, and detection kept only
  display controllers, so v0.3.0's note that gfx942 and gfx950 Instinct parts are
  eligible on Linux did not hold for any of them. They are now found, along with their
  virtual functions. Only AMD graphics devices (PCI vendor 1002) count: other processing
  accelerators, such as Intel NPUs or AMD Pensando DPUs, are still ignored. As for other
  Instinct parts, the GFX target comes from the kernel's ROCm driver (see Changed), or
  from `rocminfo` or `rocm_agent_enumerator` where it lists none, and Linux ROCm needs
  one of the supported Ubuntu releases. The first launch after updating installs ROCm
  and keeps it if it passes validation.
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
- On Debian 13, and on distributions built on Ubuntu 24.04 such as Linux Mint 22 and
  Pop!_OS, EMAP-SSN Tools never started. Its launcher checked for `libasound2` and
  `libcups2`, which those systems carry only as `libasound2t64` and `libcups2t64`, so
  the check failed on every launch, even after running the `sudo apt install` command
  it printed: the new names were used only when `/etc/os-release` said `ID=ubuntu`.
  Either name now satisfies the check, and a missing library is suggested under the
  name apt offers. The check also covers the libraries Qt itself needs (OpenGL,
  Fontconfig, FreeType and GLib, plus libxfixes3 and libxkbfile1 for QtWebEngine):
  desktop systems always have them, but on a minimal Debian 13 the check passed and
  then PySide6 failed to import with "libglib-2.0.so.0: cannot open shared object
  file". The message EMAP-SSN Tools shows when QtWebEngine cannot load now gives the
  package names for Debian 12 and 13 and Ubuntu-based distributions too, not only
  Ubuntu's.
- On Linux and macOS, `install.sh` left the four launcher scripts in `src/bin` listed as
  modified by `git status`: it makes them executable, and the repository stored them
  without that permission. A later `git pull` that changed one of them stopped with
  "Your local changes to the following files would be overwritten by merge". The
  repository now stores them executable, so the installer changes nothing git tracks,
  and on macOS `git status` no longer lists the `EMAP-SSN.app` and `EMAP-SSN Tools.app`
  bundles it creates. In a checkout where `install.sh` has already run, `git pull` of
  this update stops in the same way: run `git checkout -- src/bin/` first, which only
  undoes the permission changes.
- The readiness check that runs before each launch (`Install_Dependencies.py
  --check-only`) printed "Installed package consistency" twice: once with the expected
  esm deviations, then a plain "passed" that stood for the ESM import check after it.
  Package consistency is now reported once, and the last line reads "ESM/Transformers
  import validation: passed".
- An NVIDIA Blackwell GPU (RTX 50 series and other compute capability 10.x or 12.x
  cards) with a driver older than 580 got the CUDA 12.6 build, which has no Blackwell
  kernels: setup downloaded it, failed validation and moved to the CPU with only a
  validation error. Setup now leaves such a GPU out, prints a warning to update the
  NVIDIA driver to 580 or newer, and uses the CPU until then; after the update, the
  next launch installs CUDA 13.2. Whenever a GPU build fails to install or validate,
  setup now ends with a warning naming the build, the error (for a failed tensor check,
  each GPU's own error), and the `--refresh-backend` command that tries the GPU again.
- An NVIDIA GPU below compute capability 5.0, such as a GTX 780, or one whose driver is
  older than CUDA 12 needs (525.60.13 on Linux, 528.33 on Windows) got the CUDA 12.6
  build, which failed validation after a download of several gigabytes. Setup now leaves
  such a GPU out, says why or which driver it needs, and uses the CPU.
- When setup fell back to the CPU, the reason it printed came from whichever device was
  listed first, such as an integrated Intel GPU beside an AMD card the distribution kept
  from ROCm, or told how a GPU's target was found rather than why it was refused. It now
  gives the refusal of the GPU that would have been used.
- While a computer's GPU was not visible (unplugged, or its driver not loaded), every
  launch printed "State mismatch" lines although the installed GPU build was rightly
  kept. Only a mismatch that stops the launch is listed now.
- The bundled Tabulator stylesheet is upstream's again: one value in a print-table rule
  had been changed by accident (16px became 166px).
- In a Windows checkout, `sha256sum -c src/resources/fonts/desktop/SHA256SUMS` failed
  for every font, because the file got CRLF line ends. It now keeps LF everywhere.
- On a computer with an NVIDIA GPU older than Turing (compute capability below 7.5),
  such as a GTX 1080, and one newer than Hopper, such as an RTX 5090, setup installed
  CUDA 12.6 for both, because CUDA 13.2 does not support the older GPU. CUDA 12.6 has
  no kernels for the newer GPU, which then went unused without a word. Setup now
  leaves it out and prints a warning saying why.
- A hand-edited or damaged `ssn_backend.json` (the installer's state file in `.venv`)
  whose `schema` was not a number, such as `null`, `"abc"` or `NaN`, made every device
  lookup fail with `TypeError`, `ValueError` or `OverflowError`: the GUI's device
  options, embedding, alignment and layout generation. Such a file now counts as
  unvalidated, as an unreadable one already did, and every visible device is offered.
- On Windows, an AMD graphics card that GPU detection did not recognize took the GFX
  target of the processor's own graphics whenever the processor's name gave one: an RX
  7600S beside a Ryzen 9 7940HS w/ Radeon 780M Graphics, or a Radeon PRO W7600 beside
  a Ryzen 7 8700G, became gfx1103 integrated graphics. ROCm was then installed for the
  card with the 780M's build, which has no kernels for it, and validation could pass
  on the 780M instead. A GPU's own name now decides its target. The processor's name
  is used only for a generically named GPU such as "AMD Radeon(TM) Graphics", which is
  how Windows may name the processor's graphics. A card GPU detection does not
  recognize is now a discrete GPU without a target and gets no ROCm, though the
  processor's graphics still can; the RX 7600S, RX 7600M, RX 7600M XT, PRO W7600 and
  PRO W7500 are now recognized (see Changed). On such a PC, the first launch after
  updating installs the ROCm build for a now-recognized card, and otherwise keeps the
  installed backend once it validates.
- On Windows 11 25H2, some AMD processors whose graphics AMD's ROCm 7.14.1 list
  covers never used ROCm. Those with Radeon 760M or 740M graphics, such as the Ryzen 5
  220, 230 and 240 and the Ryzen 3 210, and the Ryzen AI 5 330 with its Radeon 820M,
  used the CPU, because GPU detection knew only the Radeon 780M among these. The Ryzen
  AI 5 430, 435 and PRO 435 and the Ryzen AI 7 445 got the ROCm build for gfx1152,
  because their Radeon 840M has the same name as the graphics of gfx1152 processors
  such as the Ryzen AI 5 340; it failed validation and they used the CPU. They are
  gfx1153, which Windows ROCm now covers, and for a Radeon 840M the processor's name
  now decides the target. On such a PC, the first launch after updating installs ROCm
  and keeps it if it passes validation.
- Intel Arc Pro graphics were not recognized and used the CPU: the Arc Pro A30M,
  A40/A50, A60 and A60M, the Arc Pro B50, B60, B65 and B70, and Arrow Lake's Arc Pro
  130T/140T. They are the same GPUs as the Arc models the XPU backend already used, and
  are now eligible for it on the same operating systems.
- GPU detection classed an Intel GPU without XPU kernels, such as UHD Graphics 770 or
  Iris Xe Graphics, as `"kind": "unknown"` rather than `"integrated"`. Intel's discrete
  GPUs (Iris Xe MAX, Arc, Data Center GPU Flex and Max) are now recognized by PCI device
  ID, and every other Intel GPU is integrated; without a device ID, the name decides.
  Because the class is part of the saved hardware profile, the first launch after
  updating on a computer with such a GPU re-validates the installed PyTorch backend once
  without reinstalling it.
- The Embedding Database Search (`Embedding_SSEARCH.py`) and the Pairwise Embedding
  Alignment (`Embedding_PWA.py`) imported `Generate_Embeddings.py` to find a model's
  plugin, and that import read the shared `tools_settings.json`, in the tool and in each
  of its worker processes, even when the tool ran from an exported settings file or an
  MCP job's settings. The plugin lookup now lives in `tools/tool_helpers/Model_Plugins.py`,
  so no tool imports another.
- The Substitution Matrix Alignment (`Align_Substitution_Matrix.py`) defined
  `E_VALUE_CUTOFF`, `MAX_TARGET_SEQS` and `COMP_BASED_STATS` after the block that applies
  saved settings when a worker process starts, so its BLASTP workers always used 1,000,000
  target sequences and composition-based statistics 2, whatever was saved. The three
  settings are now defined before that block.
- The Embedding Database Search wrote its text report (`Report_*.txt`) and its hit FASTA
  (`Hits_*.fasta`) in Windows' default encoding, so a header with a character outside it,
  such as a Greek letter, stopped the search before the Excel report was written. Both
  files are now UTF-8.
- The Tools window's help pane wraps its text to the pane. A help page kept a 600 px
  minimum width, so in a narrower pane every line ran past the right edge and had to be
  scrolled sideways. A table wider than the pane now scrolls on its own, and long names
  such as a script's break onto the next line.
- The VR Configuration window's Visual Effects tab no longer scrolls sideways at the
  launch size. Its VR client settings stand in two halves side by side only while both
  fit; in a narrower window the second half goes under the first, its fields lined up
  with the ones above. The wide layout is unchanged.
- The Configuration window's Simulation & Physics tab cut the last pixel column off "Min
  % Drop Threshold:" on Linux. The four labels beside the paired fields shared a width
  taken from their text's advance, and Linux font hinting can draw a text a pixel past
  it. They now get the width the labels ask for, as the label column does; on Windows
  the widths are unchanged. VR Configuration sizes the same labels the same way.
- The bundled interface font, Noto Sans, reached only some widgets. On Linux, labels and
  buttons kept the desktop's font, such as Ubuntu Sans, while fields, dropdowns and tabs
  drew Noto Sans. Right-click menus, message boxes and tooltips kept the system font on
  Linux and Windows (Segoe UI there). Qt gives those kinds of widget fonts of their own
  from the desktop theme, and applying the window theme brought them back after the
  bundled font was set. Every widget in Config, VR Configuration, Tools and the Viewer
  now draws the bundled fonts, at the size the desktop gives it. The windows' layout is
  unchanged on Windows; on Linux, labels and buttons take Noto Sans's widths.
- The Agent page could show light for a moment before turning dark, when dark mode had
  been chosen: it set its theme only once the page had loaded. Like the Meta Data page,
  it now sets the theme before the page is drawn.
- Launching the Viewer or the VR viewer could leave its one-time settings file
  (`ssn_viewer_*.json`) in the temporary folder. The Viewer deleted it only once the
  settings were accepted, the layout generator only after the new layout was
  generated, and the VR viewer only once its server started. So a rejected launch, a
  failed generation, a window closed during generation, or a VR launch refused because
  another VR viewer was running left the file behind. Each now deletes the file as soon
  as it has read it; the VR layout launcher does the same for its layout settings file.
- A Python command script that MCP or the Agent page started through `run` kept running
  after the Viewer quit. Quitting the Viewer now ends the script and every process it
  started; `run stop` ends them while the Viewer runs.
- `export` wrote two groups whose names differ only in letter case, such as "Alpha" and
  "alpha", to one file wherever the destination folder ignores letter case, so one
  subset silently replaced the other: on Windows, on a Mac's usual APFS volume, and on
  Linux for FAT- or NTFS-formatted drives and Windows shares. It now asks the
  destination folder, by making and removing one empty file there, and refuses whenever
  the folder ignores case. A folder that keeps case apart, such as a case-sensitive APFS
  volume, gets both files. The VR viewer's `export` refuses such names the same way.
- On a Mac, Host Cache (GiB) set to AUTO in Dynamic Programming Embedding Alignment and
  Network Injection always came to 0, so embeddings were read again from the HDF5 file
  instead of kept in RAM: the memory check asked for a figure macOS doesn't provide. On
  Linux it counted only free memory, leaving out the page cache the system can reclaim,
  so the budget came out smaller than it should. Both now read the available memory
  through psutil, which gives the figure Windows already used. On Apple silicon the GPU
  memory plan now also stays within that figure, as it was meant to.
- A long file name could give the Inputs & Outputs tab of Config and VR Config a
  horizontal scroll bar at the window's default size. The dropdowns that list files or
  profiles (Sequence Set, Network Edges, MSA, the Target Cache's file and each tab's
  profile) kept the width of their longest entry when the window opened: with
  `benchmark_sequences_sanitized_[esm2_t6_8m]_network.h5` in the network folder, the
  tab's content needed 653 px at its narrowest instead of 420 in this release. They now
  keep twelve characters at their narrowest, whatever they list; the other dropdowns are
  unchanged.

### Removed

- The **Histogram** button of Config and VR Config, and the score histogram window it
  opened: the Network Statistics Report's table now draws the same distribution.
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

[Unreleased]: https://github.com/Xuebin-Feng/EMAP-SSN/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Xuebin-Feng/EMAP-SSN/releases/tag/v0.4.0
[0.3.0]: https://github.com/Xuebin-Feng/EMAP-SSN/releases/tag/v0.3.0
[0.2.0]: https://github.com/Xuebin-Feng/EMAP-SSN/releases/tag/v0.2.0
[0.1.0]: https://github.com/Xuebin-Feng/EMAP-SSN/releases/tag/v0.1.0
