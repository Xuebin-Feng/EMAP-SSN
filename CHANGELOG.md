# Changelog

All notable changes to EMAP-SSN are documented in this file. The project uses
[Semantic Versioning](https://semver.org/); interfaces and persisted formats may
still change before version 1.0.0.

## [0.3.0] - 2026-09-28

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
  it with Python 3.13 and reinstalls every dependency, including PyTorch.
- **Restart Viewers that were running during the upgrade.** The MCP server now asks a
  Viewer whether it supports alignment snapshots and node projection. A Viewer started
  before the upgrade does not, so `get_summary` with `include_alignment`,
  `get_residue_distribution`, and `query_nodes` with `fields` fail with "upgrade and
  restart the Viewer" until it is restarted.

### Added

- 2D or 3D layout generation for the physics and UMAP engines through
  `LAYOUT_DIMENSIONS` (`2` or `3`, default `2`) in layout settings documents,
  `Layout_Cache_Generator.py`, and `emapssn_pipeline(action="start_layout_job")`.
  3D caches are stored in separate folders with a `_3D` suffix, record `physics_3d`
  or `umap_3d` as their layout mode, and are validated as `(node_count, 3)`
  coordinates. The desktop Viewer, and Viewer settings validated through MCP, reject
  a 3D cache with a message that names it as one and points to its 2D cache or
  `LAYOUT_DIMENSIONS=2`.
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
  against the SHA-256 pinned in `opt_vr/player_release.json`. See `opt_vr/README.md`.
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
  name, so residue tokens such as `C53` and `X2` select residues, and `color` scales
  take the trailing-`x` form. Command help, Viewer command metadata, and the agent's
  system prompt describe the new syntax.
- Initial node metadata is created during layout-cache generation instead of when the
  Viewer opens a cache; the Viewer only orders what the cache provides. As in v0.2.0,
  a stored metadata group without `Length` counts as an intentional column deletion.
- Integral float metadata values display and export without trailing decimals, and
  the `meta show` HUD readout uses the same formatter.
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
- `select <EXPRESSION> save` raised an internal error, because `save` was accepted as
  a mode after an expression. `save` is only valid as `select save <FILENAME>`, and
  other placements now fail with a message saying so.
- `spectrum` treated infinite metadata values as numbers. A single `inf` stretched the
  color range to infinity, so every other node took the lowest color and the
  infinite node became transparent. Infinite values are now colored gray and counted
  as invalid, like NaN.
- `color` accepted negative, NaN, and infinite scales (in v0.2.0's syntax, `x-2` or
  `xnan`). Such scales, for example `-2x`, `nanx`, or `1e400x`, are now rejected
  before any change is applied; `0x` remains valid.
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

[0.3.0]: https://github.com/Xuebin-Feng/EMAP-SSN/releases/tag/v0.3.0
[0.2.0]: https://github.com/Xuebin-Feng/EMAP-SSN/releases/tag/v0.2.0
[0.1.0]: https://github.com/Xuebin-Feng/EMAP-SSN/releases/tag/v0.1.0
