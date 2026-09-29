# Third-Party Licenses

This project is distributed under the **Apache License, Version 2.0** (see
[LICENSE](LICENSE), and [NOTICE](NOTICE) for required attributions). This file
inventories third-party material that is either bundled in this repository or
required at runtime, together with the licenses that govern it. Each such
component remains under its own license.

Last reviewed: 2026-09-28

---

## 1. Bundled code (redistributed in this repository)

These upstream distribution files are shipped inside this repository. Their
license text is included alongside them, and their copyright notices must be
preserved in any redistribution. Where noted, the project prepends a short
attribution banner; the underlying upstream body is otherwise unchanged.

| Component | Version | License | Location | License text |
|---|---|---|---|---|
| [Mol*](https://github.com/molstar/molstar) | 5.10.1 | MIT | `src/resources/esmfold/molstar.js`, `molstar.css` | [`LICENSE.molstar`](src/resources/esmfold/LICENSE.molstar) |
| [Tabulator](https://github.com/olifolkerd/tabulator) | 6.2.1 | MIT | `src/resources/meta/tabulator.min.js`, `tabulator.min.css` | [`LICENSE.tabulator`](src/resources/meta/LICENSE.tabulator) |
| [marked](https://github.com/markedjs/marked) | 18.0.9 | MIT | `src/resources/agent/marked.umd.js` | [`LICENSE.marked`](src/resources/agent/LICENSE.marked) |
| [KaTeX](https://github.com/KaTeX/KaTeX) | 0.16.8 | MIT | `src/resources/katex.min.css`, `katex.min.js`, `katex-auto-render.min.js`, `fonts/KaTeX_*.woff2` | [`LICENSE.katex`](src/resources/LICENSE.katex) |
| [Noto fonts](https://github.com/notofonts/notofonts.github.io) | Monthly 2026.05.01; Google Fonts web builds | SIL OFL 1.1 | `src/resources/fonts/desktop/noto/`, `src/resources/fonts/Noto*.woff2`, and `docs/fonts/Noto*.woff2` | [`src`](src/resources/fonts/LICENSE.Noto), [`docs`](docs/fonts/LICENSE.Noto) |

- Mol*: Copyright (c) 2017 - now, Mol* contributors. Vendored 2026-07-15 from
  the official npm package `molstar@5.10.1`, whose version is embedded in the
  JavaScript bundle. After removing only this project's three-line attribution
  banner, both files match the official jsDelivr npm artifacts byte-for-byte:
  `molstar.js` SHA-256 `5567eb19fa8e7a7b3b161d4b96807c4db244cecd2f3e7c87f99c052b8b5b5b30`;
  `molstar.css` SHA-256 `5b68ceb6d3642549b4e9b2c071e58e41b98a5350ae269180587b39da86925d55`.
- Tabulator: Copyright (c) 2015-2024 Oli Folkerd. Vendored 2026-07-04 and moved
  unchanged to `src/resources/meta/` on 2026-07-07.
- marked: Copyright (c) 2018+, MarkedJS; Copyright (c) 2011-2018, Christopher
  Jeffrey. Vendored 2026-08-06. **Version note:** the page previously loaded an
  unpinned `cdn.jsdelivr.net/npm/marked/marked.min.js`, which jsDelivr was
  silently resolving to 15.0.12 — `marked.min.js` no longer exists in 18.x, so
  the CDN fell back to the newest release still containing that filename. The
  bundled copy is now 18.0.9, which ships as `lib/marked.umd.js` (there is no
  minified UMD build; the file is already compact). The UMD build defines the
  same global `marked` object, and `setOptions()` / `parse()` are unchanged.
  Rendering was diffed between 15.0.12 and 18.0.9 across headings, GFM tables,
  fenced code, line breaks, nested lists, links with inline HTML, strikethrough
  and task lists using this project's options (`gfm: true, breaks: true`) —
  output was byte-identical in every case.
- KaTeX: Copyright (c) 2013-2020 Khan Academy and other contributors. Vendored
  2026-08-06. Only the `woff2` fonts are bundled; `katex.min.css` lists `woff2`
  first in every `@font-face`, so browsers never request the `woff`/`ttf`
  fallbacks. The fonts live in `src/resources/fonts/` because the CSS resolves
  them relative to its own location.
- Noto: vendored 2026-08-09 as a compact offline desktop core. The monthly
  release contributes the Regular, Medium, SemiBold, and Bold hinted static
  TTFs for Noto Sans and Noto Sans Mono. These eight runtime files total
  4,908,576 bytes and cover Latin, Greek, Cyrillic, IPA, combining marks, and
  common scientific punctuation. They live only under
  `src/resources/fonts/desktop/noto/`, are registered privately by Qt at
  startup, and are never installed into the operating system. Other scripts
  and emoji use fonts installed by the user in the operating system. VisPy
  registers the `NotoSans` and `NotoSansMono` regular/bold faces because its
  text visual accepts one face rather than Qt-style multi-family fallback.
  Exact filenames and SHA-256 hashes are recorded in
  `src/resources/fonts/desktop/SHA256SUMS`; release provenance and source hashes
  are recorded in `src/resources/fonts/LICENSE.fonts`.

  The embedded web surfaces use 15 variable `woff2` subsets for Noto Sans and
  Noto Sans Mono, acquired from Google Fonts CSS with all remote URLs rewritten
  to local paths in `fonts.css`. This compact 757,252-byte web set is duplicated
  under `src/resources/fonts/` and `docs/fonts/`: the local web server uses the
  first copy, while `docs/list_of_commands.html` and `docs/quickstart.html` use
  the second so standalone documentation remains self-contained. Each location includes the OFL text in
  `LICENSE.Noto`; the two `fonts.css` files are identical and must be regenerated
  together. The 4.68 MiB desktop core is not duplicated under `docs/`.

  The KaTeX fonts are **not** duplicated into `docs/` — they are MIT rather than
  OFL, and the documentation page does not use KaTeX.

The bodies of Mol*, Tabulator, marked, and KaTeX are upstream distribution
artifacts. Mol* and Tabulator are not byte-for-byte unmodified because this
project prepends attribution banners so their notices travel with the files if
copied out. Only `fonts.css` is project-generated rather than an upstream
artifact; it rewrites remote font URLs to local paths.

## 2. Remotely loaded assets

**None.** As of 2026-08-06 the user interface fetches no third-party assets at
runtime. marked, KaTeX and the two font families were previously loaded from
`cdn.jsdelivr.net`, `fonts.googleapis.com` and `fonts.gstatic.com` on every
launch — QtWebEngine's default profile is off-the-record with a memory-only HTTP
cache, so nothing persisted between runs. All four are now vendored (section 1),
which makes the UI render identically offline and stops disclosing user IP
addresses to Google on every page open.

They are served locally as follows:

| Asset | Served by |
|---|---|
| `marked.umd.js` | `Web_Server.py` static route `/agent_resource/` |
| `fonts.css`, `*.woff2` | `Web_Server.py` static route `/fonts/` |
| KaTeX CSS/JS | `EMAPSSN_Tools.py`, via a `file://` baseUrl on `setHtml()` |
| Fonts in `docs/list_of_commands.html` and `docs/quickstart.html` | relative path, pages are opened from disk |

Remaining network use is by design rather than asset loading: model weights are
downloaded from Hugging Face on first use and cached, the ESM C 6B embedding and
ESM3 Large structure backends call remote inference APIs, and the agent calls
whichever LLM endpoint is configured (the Ollama, LM Studio and llama.cpp
options are fully local). Setting up the environment also downloads software:
the launchers install `uv` from `astral.sh` when it is not already present, `uv`
provides a managed CPython 3.13 when no suitable interpreter is found, and the
installer fetches Python packages from PyPI and the selected PyTorch build from
`download.pytorch.org` or AMD's `repo.amd.com` index.

## 3. Python dependencies

Installed at runtime via `pip`/`uv`; **none of these packages are redistributed
by this project.** Earlier releases bundled an ESM wheel and a forked
Transformers wheel; both are now installed from PyPI, so section 1 no longer
lists any redistributed Python distribution. As of the PySide6 and
graspologic-native migrations there are **no strong-copyleft dependencies
remaining**; what is left is weak/file-level copyleft, which imposes
obligations only on those packages' own files. This set is compatible with the
project's Apache-2.0 license.

PySide6 is used under its **LGPL-3.0** option and is installed separately; Qt is
not included in this source repository. A normal `pip install` keeps the Qt
libraries separate and replaceable. Any future executable or installer that
redistributes Qt binaries needs a separate LGPL compliance review.

### Direct dependencies

| Package | License |
|---|---|
| PySide6 (Essentials + Addons) | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only — used under **LGPL-3.0** |
| shiboken6 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only — used under **LGPL-3.0** |
| graspologic-native | MIT (Copyright (c) Microsoft Corporation) |
| biopython | Biopython License (BSD-style) |
| esm 3.4.1.post1 | MIT (Chan Zuckerberg Biohub) — installed from PyPI, not redistributed |
| transformers 5.17.0 | Apache-2.0 (Hugging Face, upstream) — installed from PyPI, not redistributed |
| huggingface-hub, tokenizers (installed with Transformers; imported directly) | Apache-2.0 |
| torch 2.12.0 | BSD-3-Clause — one build selected at installation: CPU, CUDA 12.6, CUDA 13.2, or Intel XPU from `download.pytorch.org`; ROCm 7.14 for Windows or Linux from AMD's multi-arch index; or Apple MPS from PyPI |
| AMD ROCm 7.14 runtime wheels (`rocm`, `rocm-sdk-core`, and the per-GPU `rocm-sdk-device-*` and `amd-torch-device-*` packages) | AMD's packaging code is MIT: every source file of the `rocm` 7.14.1 meta-package carries `SPDX-License-Identifier: MIT` (Copyright Advanced Micro Devices, Inc.), although its package metadata declares no license. The runtime binaries in the other wheels remain under the terms AMD ships inside them and were not reviewed here. Downloaded dynamically from AMD's official index only when a supported AMD GPU is detected on Windows 11 25H2 or newer, or on Linux |
| mcp 2.2.0 (official Model Context Protocol Python SDK) | MIT |
| pydantic (installed with mcp; imported directly by the MCP server) | MIT |
| sentencepiece 0.2.2 | Apache-2.0 |
| protobuf 7.36.2 (required by Transformers to convert sentencepiece tokenizers) | BSD-3-Clause |
| numpy, scipy, pandas, scikit-learn | BSD-3-Clause |
| networkx, vispy, httpx, h5py, markdown, psutil | BSD-3-Clause |
| numba | BSD-2-Clause |
| umap-learn | BSD-3-Clause |
| matplotlib | Matplotlib License (PSF-based, BSD-compatible) |
| logomaker, markov-clustering, openpyxl, jsonschema | MIT |
| tqdm | MPL-2.0 AND MIT |

The numpy wheel declares `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` for
its bundled components. The numpy and scipy wheels also ship OpenBLAS with a
statically linked GCC runtime under `GPL-3.0-or-later WITH GCC-exception-3.1`;
the runtime library exception means that component imposes no copyleft
obligation.

### Accelerator runtime libraries

Accelerator builds of torch bring vendor runtime libraries that remain under their
vendors' terms. They are downloaded only for the backend selected at installation
and are not redistributed by this project.

- **CUDA 12.6 / 13.2:** NVIDIA CUDA and cuDNN libraries (cuBLAS, cuDNN, cuFFT,
  cuRAND, cuSOLVER, cuSPARSE, NVRTC, and others). On Windows they ship inside the
  torch wheel's `torch/lib`; on Linux they arrive as separate `nvidia-*` wheels.
  They are proprietary and governed by NVIDIA's license agreements for those
  components.
- **Intel XPU:** Intel oneAPI runtime wheels, for example the SYCL and compiler
  runtimes, oneMKL, and TBB, mostly under Intel's proprietary terms such as the
  Intel Simplified Software License and the Intel End User License Agreement for
  Developer Tools.
- **ROCm 7.14:** the AMD runtime wheel components listed in the table above.

### Notable transitive dependencies with copyleft terms

These impose obligations only on their own files, not on this project's code.
They are listed for completeness.

| Package | License | Pulled in by |
|---|---|---|
| certifi | MPL-2.0 | requests, httpx, httpcore, jedi |
| biotraj | LGPL-2.1-or-later | biotite (an ESM runtime dependency) |
| pywin32 (its `adodbapi` module; the package itself is PSF-licensed) | LGPL-2.1 | mcp, psutil (Windows only) |
| setuptools (its vendored `autocommand`; the package itself is MIT) | LGPL-3.0 | torch, ipython, pillow, psutil |
| botocore (its bundled CA root certificates; the package itself is Apache-2.0) | MPL-2.0 | boto3 (an ESM runtime dependency) |

All remaining transitive dependencies (safetensors, regex, filelock, sympy,
pillow, requests, joblib, pynndescent, fonttools, threadpoolctl, llvmlite, and
others) are under permissive licenses: MIT, MIT-0, MIT-CMU, BSD, ISC,
Apache-2.0 (including Apache-2.0 WITH LLVM-exception), or PSF/CNRI-Python.

`esm` is installed with `--no-deps`, so its runtime dependencies are the ones this
project declares in `src/esm_runtime_requirements.txt`, all permissive: biotite,
RDKit, ipython, and ipywidgets (BSD-3-Clause); msgpack-numpy and zstd (BSD);
boto3, pygtrie, tenacity, huggingface-hub, safetensors, and accelerate
(Apache-2.0); pydssp, py3dmol, dna-features-viewer, einops, cloudpathlib, brotli,
and attrs (MIT); plus biopython, scikit-learn, pandas, and httpx, listed above.
The ESMFold2-only NVIDIA `cuequivariance` kernel packages that `esm` declares for
Linux x86_64 are deliberately not installed.

## 4. External programs invoked as subprocesses

These are **not** bundled or redistributed. They are located on the user's system
and executed as separate processes, exchanging data through files and standard
streams. They are independent works and do not form a combined work with this
program.

| Program | License | Used by |
|---|---|---|
| NCBI BLAST+ (`makeblastdb`, `blastp`) | Public domain (US Government work) | `src/tools/Align_Substitution_Matrix.py` |
| uv (Astral) | MIT OR Apache-2.0 | the launchers and `src/Install_Dependencies.py`, which create the managed environment and install packages; the launchers install `uv` from `astral.sh` when it is missing |

Users must obtain BLAST+ separately and comply with its terms.

Hardware detection in `src/Detect_GPU.py` also queries tools that are already
present on the user's system when available: `nvidia-smi`,
`rocm_agent_enumerator`/`rocminfo`, `lspci`, `system_profiler`, and PowerShell
CIM queries. They are neither bundled nor required.

MAFFT, MUSCLE, Clustal Omega, T-Coffee, and SSEARCH/FASTA36 are mentioned in the
documentation as optional external tools whose output a user may prepare
independently, or as points of comparison. This project does not invoke, bundle,
or depend on them, and `src/tools/Embedding_MSA.py` and
`src/tools/Embedding_SSEARCH.py` implement their own embedding-based algorithms.
No license obligation arises from these mentions.

## 5. Model weights and non-open components

Model weights are **not** redistributed by this project. They are downloaded at
runtime from Hugging Face or accessed through remote APIs and remain governed by
their publishers' terms. For models with explicit usage restrictions, the
application may require a local acknowledgement before accessing model files;
that acknowledgement is not a sublicense or a substitute for reading the terms.

| Model family | Source | License |
|---|---|---|
| ESM-2 (8M, 35M, 150M, 650M) | `facebook/esm2_t6_8M_UR50D`, `esm2_t12_35M_UR50D`, `esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D` | MIT |
| ESM C 300M | `biohub/ESMC-300M` | MIT (the card also lists `other`, linked to Biohub's `THIRD_PARTY_NOTICE.md`) |
| ESM C 600M | `biohub/ESMC-600M` | MIT (the card also lists `other`, linked to Biohub's `THIRD_PARTY_NOTICE.md`) |
| ESM3-open 1.4B | `biohub/esm3-sm-open-v1` | MIT |
| ESM C 6B (remote API; `esmc-6b-2024-12`) | `https://biohub.ai` | Governed by the API provider's terms of use |
| ESM3 Large (remote API; currently `esm3-large-2024-03`) | `https://biohub.ai` | Governed by the API provider's terms of use |
| ProtBERT | [`Rostlab/prot_bert`](https://huggingface.co/Rostlab/prot_bert) | Academic Free License 3.0 (AFL-3.0). The Hugging Face repository has no license tag, but the official [`ProtTrans`](https://github.com/agemagician/ProtTrans#license) project explicitly states that its pretrained models are released under AFL-3.0. ProtTrans source code is separately MIT-licensed. |
| ProstT5 | [`Rostlab/ProstT5`](https://huggingface.co/Rostlab/ProstT5) | MIT (declared by the model repository) |
| Ankh Base, Ankh Large | [`ElnaggarLab/ankh-base`](https://huggingface.co/ElnaggarLab/ankh-base), [`ankh-large`](https://huggingface.co/ElnaggarLab/ankh-large) | CC-BY-NC-SA-4.0 — non-commercial, attribution, and ShareAlike terms; explicit application acknowledgement required before access |

> **Note on the `esm` package licensing history.** EvolutionaryScale's PyPI
> releases through 3.2.3 carry the bespoke Cambrian license. The upstream MIT
> relicense landed on 2026-05-27 at commit
> `c94ed8d763bbd7088b296949e5b401e8ea12073a`, after those releases. Releases
> through v0.2.0 therefore bundled an unmodified wheel built from that exact
> MIT-licensed commit rather than relabeling a Cambrian-licensed PyPI artifact.
> Since v0.3.0 the project installs the MIT-licensed PyPI release `esm`
> 3.4.1.post1 (Copyright 2026 Chan Zuckerberg Biohub, Inc.) with `--no-deps` and
> redistributes no `esm` code.
>
> Note that **model weights are acquired separately from Hugging Face** and are
> governed by the terms declared on their model cards at download time, which
> are currently MIT (see the table above). The non-commercial restriction that
> historically applied to ESM3-open and ESM C 600M attached to the *weights*,
> never to the code.
>
> Independently of licensing, `esm` is kept at arm's length architecturally:
> every `esm` import is lazy (inside a function or a `try`/`except`), the
> protein-language-model plugins are discovered by static AST parsing without
> importing model dependencies, and the `esmfold` structure-prediction path runs
> in a **separate process** (`src/resources/esmfold/esmfold_worker.py`). Model
> weights remain external and are never redistributed.

## 6. Artwork

The project author confirmed on 2026-08-09 that the logos and icons in
`src/bin/logos/` and screenshots in `docs/assets/` are original or otherwise
authorized. Authority to release those works remains subject to the same
University of Toronto ownership review as the rest of the project.

## 7. Optional `opt_vr` submodule

`opt_vr` is a Git submodule that points at the separate
[EMAP-SSN-VR](https://github.com/Xuebin-Feng/EMAP-SSN-VR) repository, which
carries its own Apache-2.0 `LICENSE`. Its contents are not part of this
repository's source archives, and this inventory does not cover them. That
repository tracks no binaries. Its VR client, built with Godot Engine (MIT)
and licensed Apache-2.0, is a release asset that `install_vr.bat` or the VR
viewer downloads; each release carries the client's complete source. Its
`THIRD_PARTY_NOTICES.md`, which also ships inside every client release,
inventories Godot and the components compiled into it, all under permissive
licences apart from Mozilla's CA certificate bundle (MPL-2.0).

---

## Maintenance

Re-run this review whenever a dependency is added, a bundled asset is refreshed,
or a new model backend is introduced. When refreshing a bundled asset, copy the
upstream `LICENSE` file alongside it, record the version above, and re-apply the
attribution banner where the project uses one (Mol* and Tabulator). For Mol*,
re-verify the recorded SHA-256 values with the banner removed. Regenerate both
`fonts.css` copies together. When `ESM_VERSION` changes, re-derive
`src/esm_runtime_requirements.txt` from the new release's metadata and re-check
those licenses. When a model mapping changes, re-check the license declared on
the new model card.
