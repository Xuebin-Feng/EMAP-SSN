# Changelog

All notable changes to EMAP-SSN are documented in this file. The project uses
[Semantic Versioning](https://semver.org/); interfaces and persisted formats may
still change before version 1.0.0.

## [Unreleased]

### Highlights

- **No bundled Python wheels.** ESM and Transformers are now installed from PyPI at
  their published versions. Earlier releases redistributed an ESM 3.3.0 wheel and a
  reproducible Transformers fork labeled `4.57.6+biohub.3a8956f`, because ESM 3.3.0
  required a build carrying `transformers/models/esmc`. ESM 3.4 moved ESMC and
  ESMFold2 into the `esm` package itself, so the fork is obsolete. This removes the
  project's Python wheel-redistribution obligations entirely.
- **Python 3.13.** The managed environment target moves from 3.12 to 3.13.
- **One ROCm backend instead of four.** Windows and Linux now share a single ROCm 7.14
  profile from AMD's multi-architecture channel.

### Added

- `src/esm_runtime_requirements.txt` declaring ESM's runtime dependencies, installed
  after the accelerator-specific PyTorch build. It replaces the generated file that
  lived beside the bundled wheels, and the installer rejects it if it ever names
  `torch` or `transformers`.
- `src/UPGRADE_FOLLOW_UP.txt` recording the conflicts, verification items, and known
  ceilings carried by this dependency change.
- Linux AMD GPUs whose product name is not in the pinned model-to-GFX snapshot now
  resolve their target from `rocm_agent_enumerator`/`rocminfo` instead of being
  reported as unsupported. This makes AMD Instinct parts (`gfx908`, `gfx90a`,
  `gfx942`, `gfx950`) and RDNA1/RDNA2 consumer cards (`gfx1010`–`gfx1012`,
  `gfx1031`–`gfx1036`) eligible on Linux for the first time. The reported target is
  accepted only when exactly one supported target is present — with several distinct
  agents there is no reliable way to attribute one to a specific adapter — and only
  when the ROCm channel publishes a device package for it. Windows is unchanged and
  still relies on the name snapshot, because it has no equivalent runtime source.

### Changed

- Upgraded `esm` 3.3.0 → 3.4.1.post1 (MIT, from PyPI) and `transformers`
  `4.57.6+biohub.3a8956f` → 5.17.0 (Apache-2.0, upstream), which also raises
  `huggingface-hub` to 1.32.0 and `tokenizers` to 0.23.2.
- Pinned `torch` to 2.12.0 across every backend, including both ROCm platforms, so a
  single PyTorch version now covers the whole candidate ladder.
- Consolidated the four ROCm profiles (Windows 7.14 and 7.2.1, Linux 7.2 and 6.4)
  into one ROCm 7.14 backend serving Windows and Linux from
  `repo.amd.com/rocm/whl-multi-arch`. ROCm 7.14 covers every GFX target the retired
  profiles did, so no previously eligible GPU loses support. The retired Windows
  ROCm 7.2.1 profile was limited by AMD to Python 3.12, which is why it could not
  move to the Python 3.13 environment.
- Raised the managed virtual environment from Python 3.12 to 3.13 in all four
  launchers.
- Upgraded `numba` to 0.67.0, which raised its ceiling to `numpy<2.6` and unblocked
  `numpy` 2.5.3. Also bumped PySide6 6.11.2, biopython 1.88, matplotlib 3.11.2,
  mcp 2.2.0, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, sentencepiece 0.2.2,
  tqdm 4.70.1, and vispy 0.17.0.
- Backend state schema raised to 6 and `COMPATIBILITY_REVISION` to 7. The bundled-wheel checksum fields are gone and
  the ROCm profiles are collapsed, so schema-5 state is not comparable and is
  rebuilt on first run.
- `validate_package_consistency` now parses `uv pip check` output instead of reading
  only its exit code. Because ESM is installed with `--no-deps` against a newer
  torch and Transformers than it declares, the check always reports exactly two
  incompatibilities; both are accepted only when the installed versions are the ones
  pinned here, and any other finding still fails. An unaccounted-for or unparseable
  report fails closed.
- The ESM import smoke test now exercises `esm.models.esmc` and the generic
  Transformers API used by the pLM adapters, rather than the forked
  `transformers.models.esmc` modules that no longer exist upstream at 4.x.

### Fixed

- Windows ROCm eligibility no longer offers a retired profile on an OS build the
  surviving profile does not support. The ROCm 7.2.1 detection path remained after
  the backend consolidation, so a Windows 11 build older than 25H2 could still be
  reported eligible through it and would then be installed with the ROCm 7.14 wheel,
  which AMD publishes for 25H2 only. Windows ROCm now requires build 26200 or newer
  outright and otherwise falls through to the next accelerator or CPU.

### Removed

- The Windows ROCm 7.2.1 detection path, its GFX target set, and the AMD Software
  26.2.2 version gate. The historical `rocm721`, `rocm72`, `rocm714` and `rocm64`
  profile names are gone entirely: the detector reports a single `rocm` profile and
  the installer consumes it directly, with no alias mapping. State saved by an
  earlier install is already rejected by the schema-6 check and rebuilt.
- `_windows_amd_software_version()`, which ran a PowerShell registry query on every
  detection. It existed only to gate the retired ROCm 7.2.1 profile, so nothing
  reads it now.
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

[0.2.0]: https://github.com/Xuebin-Feng/EMAP-SSN/releases/tag/v0.2.0
[0.1.0]: https://github.com/Xuebin-Feng/EMAP-SSN/releases/tag/v0.1.0
