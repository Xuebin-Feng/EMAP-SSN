# Third-Party Licenses

This project is distributed under the **Apache License, Version 2.0** (see
[LICENSE](LICENSE), and [NOTICE](NOTICE) for required attributions). This file
inventories third-party material that is either bundled in this repository or
required at runtime, together with the licenses that govern it. Each such
component remains under its own license.

Last reviewed: 2026-08-06

---

## 1. Bundled code (redistributed in this repository)

These files are shipped verbatim inside this repository. Their license text is
included alongside them, and their copyright notices must be preserved in any
redistribution.

| Component | Version | License | Location | License text |
|---|---|---|---|---|
| [Mol*](https://github.com/molstar/molstar) | see note | MIT | `src/resources/esmfold/molstar.js`, `molstar.css` | [`LICENSE.molstar`](src/resources/esmfold/LICENSE.molstar) |
| [Tabulator](https://github.com/olifolkerd/tabulator) | 6.2.1 | MIT | `src/resources/meta/tabulator.min.js`, `tabulator.min.css` | [`LICENSE.tabulator`](src/resources/meta/LICENSE.tabulator) |
| [marked](https://github.com/markedjs/marked) | 18.0.9 | MIT | `src/resources/agent/marked.umd.js` | [`LICENSE.marked`](src/resources/agent/LICENSE.marked) |
| [KaTeX](https://github.com/KaTeX/KaTeX) | 0.16.8 | MIT | `src/resources/katex.min.css`, `katex.min.js`, `katex-auto-render.min.js`, `fonts/KaTeX_*.woff2` | [`LICENSE.katex`](src/resources/LICENSE.katex) |
| [Inter](https://github.com/rsms/inter) | Google Fonts build | SIL OFL 1.1 | `src/resources/fonts/Inter-*.woff2` **and** `docs/fonts/Inter-*.woff2` | [`src`](src/resources/fonts/LICENSE.Inter), [`docs`](docs/fonts/LICENSE.Inter) |
| [Fira Code](https://github.com/tonsky/FiraCode) | Google Fonts build | SIL OFL 1.1 | `src/resources/fonts/FiraCode-*.woff2` **and** `docs/fonts/FiraCode-*.woff2` | [`src`](src/resources/fonts/LICENSE.FiraCode), [`docs`](docs/fonts/LICENSE.FiraCode) |

- Mol*: Copyright (c) 2017 - now, Mol* contributors. Vendored 2026-07-15.
  The upstream release version is not recorded in the minified bundle; record it
  here when the bundle is next refreshed.
- Tabulator: Copyright (c) 2015-2024 Oli Folkerd. Vendored 2026-07-07.
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
- Inter and Fira Code: vendored 2026-08-06 from the Google Fonts CSS the UI
  previously fetched at runtime, with the remote URLs rewritten to local files in
  `fonts.css`. 49 `woff2` subsets in total. The OFL requires its license text to
  accompany redistributed font files; both licenses are shipped verbatim rather
  than merged, as the two texts differ slightly.

  These fonts are deliberately **duplicated in two locations**: the web UI loads
  them from `src/resources/fonts/` over the local web server, while
  `docs/list_of_commands.html` loads them from `docs/fonts/` by relative path so
  the documentation is self-contained and does not reach into the source tree.
  Each copy carries its own `LICENSE.Inter` and `LICENSE.FiraCode`, which is what
  the OFL requires of each redistributed set. The two `fonts.css` files are
  identical; regenerate both together.

  The KaTeX fonts are **not** duplicated into `docs/` — they are MIT rather than
  OFL, and the documentation page does not use KaTeX.

Mol*, Tabulator, marked, and KaTeX are included **unmodified**. Mol* and
Tabulator carry an attribution banner in the file itself so the notice travels
with the code if the file is copied out. Only `fonts.css` is generated rather
than vendored verbatim, and it is a rewrite of URLs only.

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
| `marked.min.js` | `Web_Server.py` static route `/agent_resource/` |
| `fonts.css`, `*.woff2` | `Web_Server.py` static route `/fonts/` |
| KaTeX CSS/JS | `SSN_Tools.py`, via a `file://` baseUrl on `setHtml()` |
| Fonts in `docs/list_of_commands.html` | relative path, page is opened from disk |

Remaining network use is by design rather than asset loading: model weights are
downloaded from Hugging Face on first use and cached, the ESM C 6B backend calls
a remote inference API, and the agent calls whichever LLM endpoint is configured
(the Ollama, LM Studio and llama.cpp options are fully local).

## 3. Python dependencies

Installed via `pip` from `src/requirements.txt`; not redistributed by this
project. As of the PySide6 and graspologic-native migrations there are **no
strong-copyleft dependencies remaining**; what is left is weak/file-level
copyleft, which imposes obligations only on those packages' own files. This set
is compatible with the project's Apache-2.0 license.

PySide6 is used under its **LGPL-3.0** option. That permits permissively licensed
application code provided Qt remains dynamically linked and user-replaceable — a
normal `pip install` satisfies this. Do not vendor or statically link Qt.

### Direct dependencies

| Package | License |
|---|---|
| PySide6 (Essentials + Addons) | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0 — used under **LGPL-3.0** |
| shiboken6 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0 — used under **LGPL-3.0** |
| graspologic-native | MIT (Copyright (c) Microsoft Corporation) |
| biopython | Biopython License (BSD-style) |
| esm | Cambrian Open License — **see section 5** |
| torch, torchvision, torchaudio | BSD-3-Clause |
| transformers, accelerate | Apache-2.0 |
| numpy, scipy, pandas, scikit-learn | BSD-3-Clause |
| networkx, vispy, httpx, h5py, markdown | BSD-3-Clause |
| numba | BSD-2-Clause |
| umap-learn | BSD-3-Clause |
| matplotlib | Matplotlib License (PSF-based, BSD-compatible) |
| logomaker, markov-clustering, openpyxl | MIT |
| tqdm | MPL-2.0 AND MIT |

### Notable transitive dependencies with copyleft terms

These impose obligations only on their own files, not on this project's code.
They are listed for completeness.

| Package | License | Pulled in by |
|---|---|---|
| certifi | MPL-2.0 | requests / httpx |
| biotraj | LGPL-2.1-or-later | esm |

All remaining transitive dependencies (huggingface-hub, safetensors, tokenizers,
regex, filelock, sympy, pillow, requests, joblib, pynndescent, fonttools,
threadpoolctl, and others) are MIT, BSD, Apache-2.0, or PSF licensed.

`esm` 3.2.3 additionally pulls in the following, all permissive: biotite
(BSD-3-Clause), torchtext (BSD), boto3 and pygtrie (Apache-2.0), and pydssp,
py3dmol, dna-features-viewer (MIT), plus ipywidgets (BSD-3-Clause).

## 4. External programs invoked as subprocesses

These are **not** bundled or redistributed. They are located on the user's system
and executed as separate processes, exchanging data through files and standard
streams. They are independent works and do not form a combined work with this
program.

| Program | License | Used by |
|---|---|---|
| NCBI BLAST+ (`makeblastdb`, `blastp`) | Public domain (US Government work) | `src/tools/Align_Substitution_Matrix.py` |

Users must obtain BLAST+ separately and comply with its terms.

MAFFT, MUSCLE, and SSEARCH/FASTA36 are referred to in documentation and
docstrings as points of comparison only. This project does not invoke, bundle, or
depend on them, and `src/tools/Embedding_MSA.py` and
`src/tools/Embedding_SSEARCH.py` implement their own embedding-based algorithms.
No license obligation arises from these mentions.

## 5. Model weights and non-open components

Model weights are **not** redistributed by this project. They are downloaded at
runtime from Hugging Face or accessed through remote APIs, and are governed by
their own licenses, which the user accepts directly with the model provider.

| Model family | Source | License |
|---|---|---|
| ESM-2 | `facebook/esm2_*` | MIT |
| ESM C 300M | `biohub/esmc-300m-2024-12` | MIT (card also carries a legacy `other` tag) |
| ESM C 600M | `biohub/esmc-600m-2024-12` | MIT (card also carries a legacy `other` tag) |
| ESM3-open 1.4B | `biohub/esm3-sm-open-v1` | MIT |
| ESM C 6B (remote API) | `https://biohub.ai` | Governed by the API provider's terms of use |
| ProtBERT, ProstT5 | `Rostlab/*` | Verify on the model card before relying on it |
| Ankh | `ElnaggarLab/ankh-base`, `ankh-large` | Verify on the model card before relying on it |

> **Note on the `esm` package licensing history.**
>
> EvolutionaryScale originally released ESM under a family of bespoke "Cambrian"
> licenses: the Cambrian Open License for the codebase and ESM C 300M, and the
> Cambrian **Non-Commercial** License for ESM3-open and ESM C 600M. Those
> licenses are not OSI-approved and carried a naming requirement ("Built with
> ESM"). Under the project's former GPL-3.0 license that sat awkwardly against
> section 7, which forbids layering additional restrictions onto a covered work.
> Under Apache-2.0 that particular tension does not arise: Apache-2.0 places no
> equivalent restriction on what a work may be combined with. The naming
> requirement remains an obligation on anyone who redistributes `esm` itself,
> which this project does not do.
>
> EvolutionaryScale joined Chan Zuckerberg Biohub in 2025, and the ESM stack has
> since been **relicensed to MIT**. As of this review, the upstream
> `LICENSE.md` is the MIT License (Copyright Chan Zuckerberg Biohub, Inc.), the
> upstream README states "These models are available under the MIT license", and
> the Hugging Face cards for ESM3-open, ESM C 300M, and ESM C 600M declare `mit`.
> Under MIT there is no licensing conflict and no non-commercial restriction.
>
> **Caveat — the MIT relicense has not reached any release.** Verified
> 2026-08-05 against both PyPI and the upstream git tags:
>
> | Source | `LICENSE.md` |
> |---|---|
> | PyPI wheel `esm-3.2.1.post1` | Cambrian |
> | PyPI wheel `esm-3.2.3` (latest release, 2025-10-14) | Cambrian — byte-identical to 3.2.1.post1 (sha256 `be40c772…`), no mention of MIT |
> | git tag `v3.2.1` | Cambrian |
> | git tag `v3.2.3` | Cambrian |
> | branch `main` | **MIT** (Chan Zuckerberg Biohub, Inc.) |
>
> The relicense landed on `main` on **2026-05-27** (commit "Update esm repo for
> ESMC paper (#304)"), roughly seven months after the most recent release. No
> commit states that the MIT grant applies to prior versions.
>
> Consequently, **upgrading the pin does not change the license situation.** Any
> released version of `esm`, installed from PyPI or checked out at its tag, is
> obtained under the Cambrian terms. That grant is valid; it is simply not MIT.
> A claim of MIT rights over `esm` would rest on the current `main` branch, not
> on any installed artifact. Re-check when a release later than 3.2.3 appears.
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
> importing model dependencies, and the ESMFold path runs in a **separate
> process** (`src/resources/esmfold/esmfold_worker.py`). `esm` is an optional
> runtime dependency, and the software remains functional without it via the
> ESM-2, ProtBERT, and Ankh backends.

## 6. Artwork

Logos and icons in `src/bin/logos/` and screenshots in `docs/assets/` are
original works of this project and are covered by its Apache-2.0 license.

---

## Maintenance

Re-run this review whenever a dependency is added, a bundled asset is refreshed,
or a new model backend is introduced. When refreshing a bundled asset, copy the
upstream `LICENSE` file alongside it, record the version above, and re-apply the
attribution banner.
