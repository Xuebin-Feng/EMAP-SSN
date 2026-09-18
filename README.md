# EMAP-SSN

[![Python Version](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Release](https://img.shields.io/github/v/release/Xuebin-Feng/EMAP-SSN?display_name=tag)](https://github.com/Xuebin-Feng/EMAP-SSN/releases/latest)
[![Platform](https://img.shields.io/badge/platform-windows%20%7C%20linux%20%7C%20macOS%20(Apple%20Silicon)-lightgrey.svg)](https://github.com/Xuebin-Feng/EMAP-SSN)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Framework PySide6](https://img.shields.io/badge/UI-PySide6-orange.svg)](https://doc.qt.io/qtforpython/)
[![Render VisPy](https://img.shields.io/badge/Render-VisPy-red.svg)](https://vispy.org/)

**EMAP-SSN: Embedding- and Multiple-Alignment-integrated Protein Sequence
Similarity Network Platform** is an interactive, high-performance
graphical application for generating, visualizing, and analyzing traditional
and embedding-based Sequence Similarity Networks (SSNs). By integrating
Multiple Sequence Alignments (MSAs) directly into network exploration, the
platform bridges macroscopic sequence relationships with microscopic
residue-level conservation to provide a multi-scale view of protein sequence
space.
The latest release recorded in the changelog is **v0.2.0**.
This README describes the current source checkout, which may include changes
since that release. Because EMAP-SSN remains below version 1.0, command
interfaces and persisted formats may evolve between releases. See the
[changelog](CHANGELOG.md) for release history and [citation metadata](CITATION.cff)
for how to cite the software.

---
## ⚠️ Important Note

1. **Cross-Platform Support**: Host architecture, operating-system, and accelerator support is conditional. See [Compatibility](#-compatibility) before installation; Linux and Apple Silicon macOS support remains under active development.
2. **Recommended Hardware**: A compatible **NVIDIA GPU** is highly recommended for CUDA acceleration of embeddings and layout solvers. Selected AMD and Intel devices and Apple Silicon GPUs can use the backends listed in [Compatibility](#-compatibility); every accelerator must pass runtime validation.

---

## 📸 Overview

The application streamlines the entire SSN pipeline—from generation to interactive analysis—within a single unified workflow. It supports both traditional sequence similarity methods (e.g., BLAST) and modern embedding-based language model algorithms. For an end-to-end architectural guide, complete workflow description, and visual walkthrough of each GUI, see the [Quick Start Manual](docs/quickstart.html). Beyond dynamic visual formatting, the viewer provides an interactive command console with specialized commands tailored for deep analysis of the protein sequence space (see the [Viewer Command Reference](docs/list_of_commands.html) for detailed usage and examples).

---
## 🖥️ Graphical User Interface

### 🛠️ EMAP-SSN Tools GUI

All calculations related to SSN generation are centralized in the `EMAPSSN_Tools.py` GUI. The interface is organized into intuitive tabs, each representing a distinct stage of the pipeline. It includes interactive tooltips at the bottom for parameter input fields and a script description panel on the right highlighting the function of each processing script.

![EMAP-SSN Tools GUI](docs/assets/emapssn_tools_gui.png)

Each tool card also provides **Export Setting** beside **Save & Run**. The
Directories tab controls the export location through **Setting Export
Directory**, which defaults to `Cache_Files/Exported_Settings/`. An exported file
contains the current settings for one tool and only the global directories
used by that tool. For example:

```json
{
  "DIRECTORIES": {
    "FASTA_DIR": "Input_Files/Sequence_Sets"
  },
  "Sanitize_Sequences.py": {
    "INPUT_FASTA": "example.fasta",
    "OVER_WRITE": false
  }
}
```

Run any exported configuration as the sole positional argument to its tool:

```powershell
# Windows
.\.venv\Scripts\python.exe src\tools\Sanitize_Sequences.py Cache_Files\Exported_Settings\example.json
```

```bash
# Linux or macOS
./.venv/bin/python src/tools/Sanitize_Sequences.py Cache_Files/Exported_Settings/example.json
```

The settings-file argument is resolved from the terminal's current working
directory. Directory values inside the JSON retain the GUI representation:
absolute paths remain absolute, while relative paths are resolved from the
project root. Omitting the argument preserves the GUI-compatible behavior of
reading `tools_settings.json` from the project root. Model acknowledgement remains a
separate command, for example
`python src/tools/Generate_Embeddings.py --accept-model-license MODEL_ID`.

### 🔌 Local STDIO MCP server

`src/EMAPSSN_MCP_Server.py` exposes 14 pipeline programs, layout-cache generation,
Viewer inspection, command execution, and session management to local MCP clients.
It registers exactly three workflow tools:

| MCP tool | Purpose |
| --- | --- |
| `emapssn_pipeline` | Discover tools, export and validate settings, inspect files and compute capabilities, run pipeline/layout jobs, and monitor or cancel work. |
| `emapssn_viewer_data` | Discover sessions, inspect immutable snapshots and residue distributions, read command results and logs, and capture the Viewer canvas. |
| `emapssn_viewer_control` | Export and validate Viewer settings, launch or connect to sessions, execute Viewer commands, disconnect, or close a Viewer. |

Each tool accepts `action` and an `arguments` object. Call `help` to list actions,
or `describe` to inspect one action's strict argument schema. For example, pass
this JSON to `emapssn_pipeline`:

```json
{"action": "describe", "arguments": {"action": "start_job"}}
```

Pipeline IDs are values inside `arguments`, not additional MCP tool names.
Legacy standalone MCP tool names are no longer registered; restart the server
and refresh the client's tool catalog after updating.

The server uses STDIO only: the client starts one server process and communicates through its standard input
and output. Before configuring a client, launch the Viewer or Tools through its
managed launcher so `.venv` is created or updated with the pinned MCP SDK.

Use absolute paths in client configuration. On Windows, replace `<PROJECT>`
below with the absolute repository path:

```powershell
# Codex
codex mcp add emapssn -- "<PROJECT>\.venv\Scripts\python.exe" "<PROJECT>\src\EMAPSSN_MCP_Server.py"

# Claude Code (user scope)
claude mcp add --transport stdio --scope user emapssn -- "<PROJECT>\.venv\Scripts\python.exe" "<PROJECT>\src\EMAPSSN_MCP_Server.py"
```

For VS Code, add this to the appropriate `mcp.json` file:

```json
{
  "servers": {
    "emapssn": {
      "type": "stdio",
      "command": "<PROJECT>\\.venv\\Scripts\\python.exe",
      "args": ["<PROJECT>\\src\\EMAPSSN_MCP_Server.py"]
    }
  }
}
```

Antigravity and clients using the common `mcpServers` form, including many
open-source harnesses, can use:

```json
{
  "mcpServers": {
    "emapssn": {
      "command": "<PROJECT>\\.venv\\Scripts\\python.exe",
      "args": ["<PROJECT>\\src\\EMAPSSN_MCP_Server.py"]
    }
  }
}
```

On Linux or macOS, use `<PROJECT>/.venv/bin/python` and forward-slash paths.
No environment activation or wrapper script is required. A harness with a
different configuration schema needs only the same executable and server-file
arguments over its local STDIO transport.

Pipeline action `start_job` accepts parameters with optional
directory overrides, an exported JSON document, or a path to that document.
Use pipeline action `get_compute_capabilities` for metadata-only discovery of runtime
devices and memory, and `inspect_file` for read-only structural inspection
of an explicitly selected file. Device discovery does not benchmark a workload
or establish that it will fit in memory. See [MCP settings and examples](docs/mcp_settings.md)
for the strict settings contracts and GUI export compatibility. Each client connection owns one
FIFO with one running and at most 16 pending jobs. Closing or restarting that
client terminates its running job and cancels its queued jobs; different MCP
clients do not share a queue and can start conflicting calculations. Pipeline
jobs may create or overwrite files according to their settings, and cancelling
a job does not roll back files already written.

Job responses report captured stdout/stderr, the immutable settings snapshot,
and the configured result directories. They do not guess the exact scientific
files produced by a program. MCP-owned logs and snapshots are temporary and
are removed on normal server shutdown; an abrupt crash can leave a private
`sequence_similarity_network_viewer/mcp_jobs/server-*` directory below the
operating system's temporary directory, which is safe to remove when no MCP
server is running.

Viewer-data action `list_sessions` discovers running Viewers through authenticated
local endpoints. Viewer-control action `connect_session` selects one;
`disconnect_session` leaves it running, while `close_session` explicitly stops it.
For a new session, use viewer-control `export_settings` to inherit saved preferences,
review the complete JSON, then call `validate_settings` and `start_session`.
Use `get_settings_schema` to inspect the accepted fields. Exporting writes a settings
file; validation and launch consume that explicit document without merging personal
settings again. A visible Viewer uses `mode: "normal"`; `mode: "headless"` runs
without a desktop window but still uses Qt/VisPy.
See [Viewer session settings](docs/mcp_settings.md#viewer-sessions), including the
Windows host restriction on independent subprocesses.

For analysis, capture a snapshot with viewer-data `get_summary`, then reuse its
`snapshot_id` with `describe_fields`, `create_subset`, `summarize_subset`, and
`query_nodes`. Alignment and visual data are opt-in through `include_alignment`
and `include_visual`. Use `get_residue_distribution` for residue counts and
cross-tabs from an alignment snapshot. Snapshots remain frozen after Viewer edits;
capture a new one to inspect the changed state, and follow returned cursors for
paged results.

Viewer-control `execute_commands` uses the same command portal as the web agent.
Supply a unique `submission_id` and reuse it if retrying that submission. Follow
viewer-data `get_command_request` until completion; `read_command_output` provides
printed diagnostics and `capture_view` returns the current canvas as PNG. Batches
stop on failure, and command submission alone does not mean background work has
finished. The server supplies an [agent workflow guide](src/mcp_server/Agent_Instructions.md)
during MCP initialization.

### ⚙️ EMAP-SSN Configuration GUI

The configuration GUI in `EMAPSSN_Config.py` simplifies input file selection and parameter tuning for SSN generation. The **Simulation & Physics** tab configures the SSN force-directed physics layout solver; UMAP remains available as an optional non-physics layout mode on the Inputs & Outputs tab. Each tab has a **Saved Config** selector: `(custom)` values are kept together in the project-root `viewer_settings.json`, while named per-tab JSON profiles are stored below the directory selected on the Directories tab. Its default expression is `$cache_file$/Saved_Config`, which resolves to `Cache_Files/Saved_Config/` with the default Cache File Directory. Selecting `(default)` loads read-only built-in values, and `(new)` creates a named profile from the settings currently shown. Directory values may begin with `$input_file$`, `$cache_file$`, or `$analysis_result$` to resolve beneath the corresponding configurable base directory; ordinary relative and absolute paths retain their existing behavior.

The GUI also features a **Compute Network Statistics** utility that analyzes network density and outputs a report in the right panel to guide selection of an optimal similarity cutoff. The **Consistency Check** utility compares the similarity network against the Multiple Sequence Alignment (MSA) to ensure sequence headers and indexes match across all files.

![EMAP-SSN Configuration GUI](docs/assets/emapssn_config_gui.png)

### 🔍 EMAP-SSN Viewer GUI

The main visualization window, `EMAPSSN_Viewer.py`, provides a 2D network canvas
with mouse and keyboard controls for panning, zooming, selection, and formatting.
Its in-line command console (HUD) runs analytical operations, highlights residues,
selects clusters, and exports figures. The `meta`, `esmfold`, and `agent` commands
open session-specific utilities in the system browser: a metadata table, a Mol*
structure viewer, and a conversational agent. The Tools documentation panel uses
embedded QtWebEngine.

**Layout Dimensions** in Configuration offers **2D (desktop viewer)** and
**3D (VR viewer)**. Both physics and UMAP can generate either coordinate format;
3D caches use a separate cache folder with the `_3D` suffix and are intended for
the optional `opt_vr` submodule. The desktop Viewer uses the 2D workflow.

![EMAP-SSN Viewer GUI](docs/assets/emapssn_viewer_gui.png)

---
## 🧬 System Workflow

Start with a protein FASTA file and choose a traditional BLAST or embedding-based
network. The desktop workflow is:

1. **Prepare sequences.** Run `Sanitize_Sequences.py` from Tools to standardize the
   FASTA used by downstream calculations.
2. **Build a network.** Run `Align_Substitution_Matrix.py` for BLASTP, or generate
   embeddings with `Generate_Embeddings.py` and score pairs with
   `Align_Similarity_Matrix.py`. `Parse_BLAST_Output.py` imports external BLAST
   tabular results using a matching FASTA manifest. The BLAST calculation requires
   separate `blastp` and `makeblastdb` executables; the importer reads saved results
   without running BLAST.
3. **Add an alignment when needed.** Generate an embedding-guided MSA with
   `Embedding_MSA.py`, or select an existing compatible full or sparse alignment.
   An MSA is optional for network viewing and required for residue-level analysis.
4. **Configure and cache the layout.** Select the matching node FASTA and network
   HDF5 in Configuration, choose **2D (desktop viewer)**, and select an existing
   compatible cache or **(New Layout Cache)**. For a new cache, **Save & Run**
   calculates the layout before opening the Viewer. Reopening a compatible cache
   reuses its coordinates and verified generation settings.
5. **Explore and export.** Use the Viewer console to select, color, cluster,
   inspect residues, and export figures or sequence subsets.

The default layout uses a force-directed physics solver. **UMAP Mode** offers a
manifold layout based on network scores and uses the `umap-learn` dependency
already included in the managed installation. `UMAP_NEIGHBORS` counts up to $k$
other nodes per sequence; the implementation adds a separate self-neighbor slot.
Missing connections remain missing rather than becoming zero-distance matches.
UMAP uses its neighbor topology instead of the physics layout's similarity and
top-edge filters.

![System Workflow](docs/assets/work_flow.png)

See the [Quick Start Manual](docs/quickstart.html#workflow) for a visual walkthrough
and the [Viewer Command Reference](docs/list_of_commands.html) for command syntax.

---

## 🚀 Key Features

*   **Embedding-Based Dynamic Programming Alignment**: Score residue pairs using protein-language-model embeddings for pairwise alignment, network construction, and guide-tree multiple alignment.
*   **Interactive Visualization**: PySide6 and VisPy provide a 2D canvas for panning, zooming, selecting, and formatting nodes and edges. Performance depends on the dataset and rendering hardware.
*   **Integrated Command Console (HUD)**: Execute analytical commands (such as `zoom`, `select`, `color`, `cluster`, `subcluster`, and `logo`) directly inside the viewer viewport for instant formatting and analysis.
*   **Integrated Multiple Sequence Alignments (MSA)**: Bridge macroscopic network topology with residue-level conservation. Map conservation scores directly onto nodes and extract consensus sequence details interactively.
*   **Comprehensive Utilities Suite**: Centralized GUI in `EMAPSSN_Tools.py` supporting sequence sanitization, embedding generation (ESM, ProtBERT, ProstT5), network edge filtering, guide-tree MSA generation, and sequence extraction/injection.
*   **Cross-Platform Hardware Acceleration**: Automatic eligibility checks and runtime validation for supported NVIDIA CUDA, AMD ROCm, Intel XPU, and Apple MPS configurations, with safe fallback to CPU.
*   **Scriptable Workflows**: Export tool and layout JSON for batch execution, or use the local MCP server for pipeline jobs, Viewer commands, and bounded scientific data inspection.

---

## ✅ Compatibility

*Compatibility snapshot: September 2026.* Compatibility is the intersection of
the application detector, the pinned Python packages, the operating system,
and the installed GPU driver. A device can be eligible for an installation
attempt without being guaranteed to work; the installer records it as
validated only after a tensor calculation succeeds on that device.

The status terms used below are:

- **Tested:** the application has been run on that operating-system family.
- **Supported:** the current installer and pinned upstream binary matrix cover
  the configuration.
- **Provisional:** the detector can attempt it, but the configuration has not
  been project-tested or is outside a fully validated upstream combination.
- **Unsupported:** the current managed Python 3.13 dependency set cannot be
  installed or the application deliberately excludes the configuration.

### Operating systems and processor architectures

| Platform | Status | Requirements and limits |
| --- | --- | --- |
| Windows x64 | Supported | CPU use requires Windows 10 version 1809 or newer. NVIDIA CUDA requires Windows 10 22H2 or Windows 11. Intel XPU requires Windows 11. Native AMD ROCm requires Windows 11 25H2 build 26200+, the only Windows build covered by the ROCm 7.14 profile. Windows 11 is recommended. |
| Windows x86 (32-bit) | Unsupported | The pinned binary dependencies do not provide Win32 wheels. |
| Windows ARM64 | Unsupported | Python 3.13 and `uv` exist for ARM64, but the pinned PyTorch 2.12.0 Python 3.13 CPU artifact and the CUDA, XPU, and ROCm builds used by this project do not provide a complete native ARM64 environment. Running the x64 environment under Windows emulation is not project-supported. |
| Linux x86_64 | Tested on Ubuntu and Debian | Exact tested release numbers were not recorded. The pinned Qt wheel requires glibc 2.34 or newer; Ubuntu 22.04+ and Debian 12+ satisfy that baseline. The installer automates GUI system dependencies only on Debian-family systems. |
| Other x86_64 Linux distributions | Provisional | Modern glibc-based distributions may work after their Qt/XCB/QtWebEngine packages are installed manually. Fedora, RHEL, SUSE, Arch, and other distributions have not been project-tested. Accelerator support remains subject to the narrower vendor tables below. |
| Linux ARM64 and other Linux architectures | Unsupported | Some individual upstream packages publish ARM wheels, but the complete pinned application environment has not been resolved or project-tested on these architectures. |
| macOS ARM64 | Supported | Apple Silicon only, macOS 14 or newer, using MPS or CPU. |
| macOS x86_64 | Unsupported | PyTorch 2.12.0 used by this project does not provide the required Intel macOS runtime. |

The operating-system baseline follows the pinned [Qt 6.11 platform
matrix](https://doc.qt.io/qt-6/supported-platforms.html), the [official PyTorch
wheel index](https://download.pytorch.org/whl/cpu/torch/), and the platform
requirements linked below. "Tested" refers to this application, whereas a
vendor listing alone does not constitute live application testing.

### Accelerator support by platform

| Accelerator | Windows x64 | Linux x86_64 | macOS ARM64 |
| --- | --- | --- | --- |
| CPU | Supported on Windows 10 version 1809+ and Windows 11 | Supported on the tested Ubuntu/Debian families; other distributions are provisional | Supported on macOS 14+ |
| NVIDIA CUDA | CUDA 13.2 or CUDA 12.6 on Windows 10 22H2+/Windows 11 | CUDA 13.2 or CUDA 12.6; distribution and driver must support the selected runtime | Not supported |
| AMD ROCm | Windows 11 25H2 build 26200+ only: ROCm 7.14 | Ubuntu only: ROCm 7.14 | Not supported |
| Intel XPU | Windows 11 only for supported Arc/Core Ultra Arc devices | Supported only for the device/OS combinations below | Not supported |
| Apple MPS | Not applicable | Not applicable | Supported on Apple Silicon with macOS 14+ |

#### NVIDIA CUDA

| Installer profile | Eligibility | Upstream binary coverage |
| --- | --- | --- |
| CUDA 13.2 | Every selected NVIDIA GPU must report compute capability 7.5 or newer and every driver must be version 580 or newer. | PyTorch 2.12 classifies CUDA 13.2 as experimental and publishes binaries for its listed Turing-and-newer architectures. Unlisted capability values remain provisional until runtime validation succeeds. |
| CUDA 12.6 | Used when the CUDA 13.2 conditions are not met. The effective PyTorch binary floor is compute capability 5.0. | Covers the listed Maxwell, Pascal, Volta, Turing, Ampere, Ada, and Hopper capability generations. Devices below capability 5.0 fall back to CPU. |

CUDA 12.x minor compatibility requires at least NVIDIA driver 525.60.13 on
Linux or 528.33 on Windows; a current vendor driver is recommended. See the
[PyTorch 2.12 support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md),
[current NVIDIA compute-capability table](https://developer.nvidia.com/cuda/gpus),
[legacy capability table](https://developer.nvidia.com/cuda/gpus/legacy), and
[CUDA 12.6 release notes](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html).

#### AMD ROCm

The application uses a pinned model-to-GFX snapshot and does not guess support
for newly released hardware. The following are application eligibility targets,
not a promise that every product with the same architecture will pass its
driver and tensor validation:

A single ROCm profile now covers both operating systems. Windows and Linux
install the same PyTorch 2.12.0 build from AMD's multi-architecture channel,
selected per GFX target:

| Platform/profile | Eligible GFX targets | Additional requirements |
| --- | --- | --- |
| Windows ROCm 7.14 / PyTorch 2.12.0 | `gfx1030`, `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`, `gfx1152`, `gfx1200`, `gfx1201` | Windows 11 25H2 build 26200+ and a supported AMD driver. |
| Linux ROCm 7.14 / PyTorch 2.12.0 | The Windows set, plus any other target the ROCm channel ships a device package for: `gfx908`, `gfx90a`, `gfx942`, `gfx950` (Instinct), `gfx1010`–`gfx1012` (RDNA1), and `gfx1031`–`gfx1036`, `gfx1153`, `gfx1250`. | A listed Ubuntu release, readable and writable `/dev/kfd`, and a target reported by `rocm_agent_enumerator` or `rocminfo`. |

The two platforms differ because of how the GFX target is identified. Windows
exposes no way to query it, so the target is inferred from the product name
against a pinned snapshot, and only models in that snapshot are eligible. On
Linux the installed ROCm stack reports the target directly; when the product
name is not in the snapshot the reported target is used instead, provided the
ROCm channel publishes a device package for it. That covers AMD Instinct parts
and RDNA1/RDNA2 consumer cards whose `lspci` names the snapshot never matched.
If two or more distinct targets are reported, the reported value is not
attributed to any single adapter and name matching remains the only source.

Earlier releases offered four ROCm profiles (Windows 7.14 and 7.2.1, Linux 7.2
and 6.4). ROCm 7.14 covers every GFX target the retired profiles did, so no
previously eligible GPU loses support. The retired Windows ROCm 7.2.1 profile
was limited by AMD to Python 3.12, which is why it could not move to the
managed Python 3.13 environment.

One consequence is worth stating plainly: **Windows ROCm now requires
Windows 11 25H2 build 26200 or newer.** The retired 7.2.1 profile was the only
one AMD published for earlier Windows 11 builds. A Windows 11 machine below
25H2 is no longer eligible for ROCm and uses the next available accelerator or
CPU.

The installer does not install the Linux kernel driver or system ROCm stack.
Consult the [ROCm 7.14 matrix](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)
for the matching operating system, driver, and hardware requirements.

#### Intel XPU

| Device family | Supported operating-system combinations |
| --- | --- |
| Intel Arc A-Series, Arc B-Series, and Core Ultra processors with Arc graphics | Windows 11; Ubuntu 24.04, 25.10, or 26.04 as admitted by the current application snapshot. Ubuntu 26.04 remains project-untested. |
| Core Ultra Series 3 / Panther Lake with Arc graphics | Windows 11; Ubuntu 25.10 or 26.04. Ubuntu 26.04 remains project-untested. |
| Intel Data Center GPU Max | Ubuntu 22.04, RHEL 9.2, or SLES 15. The non-Ubuntu combinations are upstream-listed but project-untested. |
| Intel HD, UHD, Iris, and non-Arc integrated graphics | Not eligible for XPU; another accelerator or CPU is used. |

Integrated Arc graphics are eligible for the XPU PyTorch installation, including
Arc 130V/140V, 130T/140T, and B370/B390 model names. After runtime tensor validation,
automatic workload benchmarks compare available devices with CPU; integrated
graphics are not excluded based on assumed performance.

The Intel driver must already be installed. See the [PyTorch Intel XPU
matrix](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) for the
validated hardware and operating-system combinations.

#### Apple MPS

MPS is selected only when both macOS and the `arm64` architecture are detected.
All Apple Silicon generations use the same eligibility rule and must pass the
runtime tensor check; otherwise the environment falls back to CPU. See Apple's
[PyTorch MPS requirements](https://developer.apple.com/metal/pytorch/).

### Backend selection and limitations

- Compatible discrete GPUs are preferred over integrated GPUs; NVIDIA, AMD,
  and Intel are ordered within the same device class.
- One backend-specific PyTorch build is installed per `.venv`. GPUs from
  different vendors cannot be used simultaneously in the same environment.
- Multiple validated NVIDIA or Intel devices can remain available to the
  application's automatic benchmark. For mixed AMD integrated/discrete
  systems, only the preferred GFX target is installed.
- A failed installation or tensor validation advances to the next eligible
  backend and ultimately to CPU.
- The selected backend and validation results are stored in
  `.venv/ssn_backend.json`. Requirements, pinned ESM and Transformers versions, compatibility
  rules, or a change to the required PyTorch profile invalidate the relevant
  saved state. Physical GPU addresses, device enumeration order, and driver
  updates that remain within the same CUDA compatibility profile do not force
  PyTorch to be reinstalled; the existing build is validated and reused.
- A shared HPC `.venv` retains its installed accelerator build when inspected
  from a CPU-only login node. CPU work remains available there, and accelerator
  tensor validation resumes on a compute node. Moving to a genuinely different
  backend profile, such as CUDA 12.6 to CUDA 13.2 or a different ROCm GFX
  target, still selects the required build.
- Run
  `python src/Install_Dependencies.py --refresh-backend` inside the managed
  environment to deliberately reinstall from the currently visible candidate
  ladder. On a CPU-only node, this explicit override selects the CPU build.

---

## ⚙️ Installation Steps

1. **Clone the repository:**
   Download the repository or clone it into a local project directory:

   ```bash
   git clone https://github.com/Xuebin-Feng/EMAP-SSN.git
   cd EMAP-SSN
   ```

2. **Set up the environment:**

   The generated Viewer and Tools launchers create a managed Python 3.13
   environment and automatically install and validate one pinned PyTorch
   backend. Review [Compatibility](#-compatibility) for supported hosts,
   accelerator profiles, drivers, and fallback behavior before continuing.

   * **🪟 Windows**:
     Double-click `install.bat` in the project root to generate Windows Shortcuts (`.lnk` files) in the project root and optionally on your Desktop.

     > [!TIP]
     > It is highly recommended to enable **Developer Mode** in your Windows Settings (Search for "Developer settings" in Windows). This allows symbolic links to be created without elevation, which is required by the Hugging Face `transformers` cache model download system to avoid duplicating file storage.

     > [!IMPORTANT]
     > **Enable long path support.** Windows limits a full path to 260 characters by default, while Linux and macOS allow far longer. Generated outputs — predicted structure `.pdb` files in particular — are named after sequence headers, so a long header inside an already-deep project folder can exceed the limit and fail to write, even though the same run succeeds on Linux or macOS.
     >
     > Run this once in an **Administrator** PowerShell, then reboot:
     > ```powershell
     > New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
     > ```
     > Alternatively, enable **Computer Configuration → Administrative Templates → System → Filesystem → "Enable Win32 long paths"** in the Group Policy Editor (`gpedit.msc`).
     >
     > If you cannot change this setting, keep the project close to the drive root (for example `C:\SSN\`) rather than nested under a long folder chain such as a synced OneDrive directory. Note that external binaries invoked by the pipeline, such as NCBI BLAST, may not be long-path aware regardless of this setting — a short project path is the most reliable option.

   * **🍏 macOS (Apple Silicon only)**:
     Open Terminal in the project root, make the installer executable, and run it:

     ```bash
     chmod +x install.sh
     ./install.sh
     ```

     The shared installer detects macOS, configures permissions for scripts in `src/bin/`, and generates Finder-native `EMAP-SSN.app` and `EMAP-SSN Tools.app` launchers in the project root.

     > [!IMPORTANT]
     > Only Apple Silicon Macs are supported. Intel-based Macs are not supported by the pinned PyTorch runtime.

   * **🐧 Linux**:
     Open a terminal in the project root, make the installer executable, and run it:

     ```bash
     chmod +x install.sh
     ./install.sh
     ```

     On Ubuntu/Debian, the installer detects missing Qt xcb and QtWebEngine system libraries and offers to install them with `sudo apt`. These libraries cannot be installed by `pip` or `uv`.

     > [!NOTE]
     > On Fedora/RHEL, install the equivalent system libraries before launching:
     > ```bash
     > sudo dnf install nss nspr libXcomposite libXdamage libXrandr libxkbcommon-x11 \
     >                  libXtst mesa-libgbm mesa-libEGL libxslt alsa-lib cups-libs
     > ```

     This will configure execution permissions and generate launchers (`emapssn` and `emapssn_tools`) as well as system `.desktop` application entries.

3. **Launch the application:**

   - **Windows:** Open `EMAP-SSN.lnk` or `EMAP-SSN Tools.lnk` from the project root or from the Desktop if you selected that installer option.
   - **macOS:** Open `EMAP-SSN.app` or `EMAP-SSN Tools.app` from the project root.
   - **Linux:** Run `./emapssn` or `./emapssn_tools`, or use the generated desktop application entries.

   `emapssn` opens the configuration GUI first. Select the input files and layout settings there, then click **Save & Run** to save the settings and launch the visualization window. On the first application launch, the managed launcher creates `.venv`, installs the pinned dependencies, and validates the selected compute backend; later launches reuse that environment while it remains compatible.

### Headless layout-cache generation

When **(New Layout Cache)** is selected in EMAP-SSN Configuration, **Export Layout
Settings** saves a generation-only JSON file. Run that file with the same
managed Python environment to calculate and save the cache without opening the
viewer:

```powershell
# Windows, from the project root
.\.venv\Scripts\python.exe -u src\Layout_Cache_Generator.py Cache_Files\Exported_Settings\example.json
```

```bash
# Linux or macOS, from the project root
./.venv/bin/python -u src/Layout_Cache_Generator.py Cache_Files/Exported_Settings/example.json
```

The export location follows **Setting Export Directory**, whose default is
`Cache_Files/Exported_Settings/`. This command generates a cache without starting
the interactive Viewer. An existing destination cache is rejected rather than
overwritten.

Layout execution JSON uses `schema_version: 2`, `kind: "layout"`, and named
`inputs`, `network`, `layout`, `simulation`, `physics`, `packing`, and `output`
sections. It contains generation settings only. Viewer JSON uses `kind: "viewer"`
and `inputs`, `alignment`, `visualization`, and `directories` sections. Score
interpretation, edge filters, UMAP settings, and box scale are read from verified
cache provenance, never repeated in Viewer JSON. Both exports inherit relevant
saved preferences; old execution JSON must be re-exported. Personal settings
and pipeline-document formats are unchanged. See [MCP settings](docs/mcp_settings.md)
for the export-first workflow and overlay format.

### Additional language fonts

The application bundles a 4.68 MiB Noto Sans/Noto Sans Mono desktop core with
Latin, Greek, Cyrillic, IPA, combining-mark, and common scientific punctuation
coverage. Qt uses fonts installed in the operating system when this core does
not contain a requested glyph.

For another writing system, download the appropriate family from the
[official Noto Fonts site](https://notofonts.github.io/) and install it normally:

- **Windows:** right-click the font file and select **Install** or **Install for
  all users**.
- **macOS:** open the font in Font Book and select **Install Font**.
- **Linux:** copy it to `~/.local/share/fonts/`, run `fc-cache -f`, and restart
  the application.

Copying a font into `src/resources/fonts/desktop/` is not sufficient: Qt does
not scan that directory, and the application registers only the files declared
in its bundled manifest. System-installed fonts become available after the
application is restarted. They can provide fallback in PySide6/Qt widgets, but
VisPy network labels remain limited to the bundled Noto Sans face.

---

## 🔧 Linux Troubleshooting

* **Blank, black, or repeatedly failing network canvas on a Wayland session.**
  The Viewer selects `QT_QPA_PLATFORM=xcb` when Wayland is detected or explicitly requested, while preserving offscreen/headless platforms. On affected Linux drivers, Qt's native Wayland plugin supplies an OpenGL ES context while VisPy compiles desktop GLSL shaders, producing repeated `Error drawing visual` messages. XWayland/XCB supplies the compatible desktop OpenGL context; the installer includes its required Ubuntu/Debian libraries.

* **The application closes immediately when launched from a desktop icon.**
  The Ubuntu/Debian launchers now check the Qt/XCB libraries before startup and print the exact `sudo apt install` command for anything missing. If any later startup step fails, a desktop-launched terminal remains open until you press Enter so the original error stays visible.

* **EMAP-SSN Tools exits immediately, or crashes inside Chromium.**
  If the documentation panel fails after the libraries above are installed — common inside containers or on hardened kernels where the Chromium sandbox cannot start — disable the sandbox:
  ```bash
  QTWEBENGINE_CHROMIUM_FLAGS=--no-sandbox ./emapssn_tools
  ```

* **`unable to lock file` when opening an HDF5 network.**
  HDF5 file locking fails on NFS and some network home directories. Set `HDF5_USE_FILE_LOCKING=FALSE` before launching.

---

## 📂 File Structure

```directory
EMAP-SSN/
│
├── install.bat               # Windows installer (creates .lnk shortcuts)
├── install.sh                # Linux/macOS installer with platform-specific launchers
│
├── src/                      # Source code directory
│   ├── EMAPSSN_Viewer.py         # Main PySide6 / VisPy desktop visualization application
│   ├── EMAPSSN_Tools.py          # GUI for preparing sequences, embeddings, networks, and alignments
│   ├── EMAPSSN_Config.py         # GUI configuration manager for inputs, thresholds, and models
│   ├── EMAPSSN_MCP_Server.py     # Three local STDIO MCP workflow entry points
│   ├── Layout_Cache_Generator.py # Standalone physics/UMAP layout-cache generation
│   │
│   ├── bin/                  # Startup scripts and launchers
│   │   ├── EMAPSSN.bat          # Windows platform startup script
│   │   ├── EMAPSSN_Tools.bat     # Windows Tools startup script
│   │   ├── EMAPSSN.sh           # Linux/macOS platform startup script
│   │   ├── EMAPSSN_Tools.sh      # Linux/macOS Tools startup script
│   │   ├── EMAPSSN_Desktop_Launcher.bat
│   │   ├── EMAPSSN_Desktop_Launcher.sh
│   │   ├── EMAPSSN_Terminal_Launcher.sh
│   │   └── logos/            # Application custom icon files (.png and .ico)
│   │
│   ├── commands/             # Command modules for interactive viewer console
│   ├── desktop/              # Qt integration, Viewer state, snapshots, and settings contracts
│   ├── mcp_server/           # Workflow dispatch, pipeline jobs, and Viewer session adapters
│   ├── resources/            # Configuration and system prompts
│   ├── tools/                # Executable processing scripts exposed by EMAPSSN_Tools
│   │   └── tool_descriptions/ # Markdown documentation displayed by the Tools GUI
│   ├── utilities/            # Focused cache, network, hardware, alignment, and FASTA helpers
│   └── web_ui/               # Embedded web UI backend and interfaces
│       └── plugins/          # Auto-discovered bundled web-utility descriptors
│
├── docs/                     # Screenshots, documentation, and metadata template
│   ├── quickstart.html       # Comprehensive Quick Start Manual & GUI architecture guide
│   ├── list_of_commands.html # Interactive Viewer command reference
│   ├── mcp_settings.md      # MCP actions, export workflows, and execution settings
│   └── web_plugin_development.md # Browser utility extension contract
│
├── tests/                    # Focused regression tests
├── opt_vr/                   # Optional VR viewer submodule
│
├── viewer_settings.json      # Settings saved by the configuration GUI
├── tools_settings.json       # Shared settings saved by the Tools GUI
│
├── Input_Files/              # Viewer and tool input files
│   ├── Sequence_Sets/        # Protein sequence sets and subsets (.fasta)
│   ├── Multiple_Alignments/  # Full or sparse multiple-sequence alignments
│   ├── Networks_EValues/    # Embedding- or BLAST-derived network files (.h5)
│   ├── Header_Lists/         # Reusable sequence-header cohorts
│   ├── Meta_Data/            # Imported and exported node metadata
│   └── Batch Scripts/        # User batch and helper scripts
│
├── Cache_Files/              # Reusable layouts and intermediate/session artifacts
│   ├── Saved_Layouts/        # Manifest-bound layout snapshots (.h5)
│   ├── Saved_Config/         # Named per-tab Configuration profiles
│   ├── Structures/           # Predicted structures and viewer assets
│   └── Exported_Settings/    # Exported tool, layout, and Viewer execution JSON
│
├── Embeddings/               # Protein-language-model embedding databases (.h5)
└── Analysis_Results/          # User-facing exported analysis results
    ├── Sequence_Export/       # Extracted sequence subsets
    ├── Alignment_Report/      # Pairwise-alignment reports
    ├── Saved_Images/          # Viewer image exports
    ├── Cluster_Label/         # Cluster and specificity reports
    └── Sequence_Logos/        # Sequence-logo graphics
```

---

## Acknowledgements

EMAP-SSN was developed by Xuebin Feng while affiliated with the University of Toronto.

## 🤝 Contributing

Contributions are welcome! Please feel free to open Issues or submit Pull Requests to enhance computational efficiency, layout performance, UI responsiveness, or commands for analyses.

Bundled browser utilities use a validated startup-discovery contract. See the
[web-utility plugin development guide](docs/web_plugin_development.md) for the
manifest, registration, activation, and failure-isolation rules.

## 📄 License

Copyright 2026 Xuebin Feng
Author affiliation: University of Toronto

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the [LICENSE](LICENSE)
file for the specific language governing permissions and limitations under the
License, and [NOTICE](NOTICE) for required attributions.

### Third-party components

This repository bundles Mol* and Tabulator, depends on Python packages under a
range of licenses, and can load protein-language-model weights governed by
their own terms. A full inventory, including which components are redistributed
and which are merely required at runtime, is in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

The managed environment also installs the official `mcp` 2.2.0 Python SDK and
its Pydantic runtime dependency; both are MIT-licensed and are not bundled in
the source repository.

The optional Ankh weights are licensed separately under **CC BY-NC-SA 4.0** and
are restricted to non-commercial use. The application labels these models and
requires explicit acknowledgement before accessing their files; the integration
code remains Apache-2.0 and does not redistribute the weights.

The GUI uses **PySide6 under its LGPL-3.0 option**. PySide6 and Qt are installed
separately and are not included in this source repository. Any future executable
or installer that redistributes Qt binaries needs a separate LGPL compliance
review.

**No Python wheels are redistributed by this repository.** Earlier releases
bundled an ESM 3.3.0 wheel and a reproducible fork of Transformers labeled
`4.57.6+biohub.3a8956f`, because ESM 3.3.0 required a Transformers build
carrying `transformers/models/esmc`. ESM 3.4 moved ESMC and ESMFold2 into the
`esm` package itself, so both are now installed from PyPI at their published
versions — MIT-licensed `esm` 3.4.1.post1 and Apache-2.0 `transformers` 5.17.0 —
and the fork is no longer needed or referenced. This removes the project's
wheel-redistribution obligations entirely.

Installation is not offline: ordinary PyPI dependencies, the selected PyTorch
build, and model weights are downloaded separately. Model weights retain their
publishers' licenses. See sections 1 and 5 of
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

Before publishing a release, confirm that copyright ownership and release
authority have been resolved and that all required technical gates pass.
