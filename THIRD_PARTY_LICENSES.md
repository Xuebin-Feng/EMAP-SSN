# Third-Party Licenses

This project is distributed under the **Apache License, Version 2.0** (see
[LICENSE](LICENSE), and [NOTICE](NOTICE) for required attributions). This file
inventories third-party material that is either bundled in this repository or
required at runtime, together with the licenses that govern it. Each such
component remains under its own license.

Last reviewed: 2026-08-09

---

## 1. Bundled code (redistributed in this repository)

These upstream distribution files are shipped inside this repository. Their
license text is included alongside them, and their copyright notices must be
preserved in any redistribution. Where noted, the project prepends a short
attribution banner; the underlying upstream body is otherwise unchanged.

| Component | Version | License | Location | License text |
|---|---|---|---|---|
| [ESM](https://github.com/evolutionaryscale/esm) | 3.3.0, source commit `c94ed8d763bbd7088b296949e5b401e8ea12073a` | MIT | `src/resources/wheels/esm-3.3.0-py3-none-any.whl` | [`LICENSE.esm`](src/resources/wheels/LICENSE.esm) |
| [Mol*](https://github.com/molstar/molstar) | 5.10.1 | MIT | `src/resources/esmfold/molstar.js`, `molstar.css` | [`LICENSE.molstar`](src/resources/esmfold/LICENSE.molstar) |
| [Tabulator](https://github.com/olifolkerd/tabulator) | 6.2.1 | MIT | `src/resources/meta/tabulator.min.js`, `tabulator.min.css` | [`LICENSE.tabulator`](src/resources/meta/LICENSE.tabulator) |
| [marked](https://github.com/markedjs/marked) | 18.0.9 | MIT | `src/resources/agent/marked.umd.js` | [`LICENSE.marked`](src/resources/agent/LICENSE.marked) |
| [KaTeX](https://github.com/KaTeX/KaTeX) | 0.16.8 | MIT | `src/resources/katex.min.css`, `katex.min.js`, `katex-auto-render.min.js`, `fonts/KaTeX_*.woff2` | [`LICENSE.katex`](src/resources/LICENSE.katex) |
| [Inter](https://github.com/rsms/inter) | Google Fonts build | SIL OFL 1.1 | `src/resources/fonts/Inter-*.woff2` **and** `docs/fonts/Inter-*.woff2` | [`src`](src/resources/fonts/LICENSE.Inter), [`docs`](docs/fonts/LICENSE.Inter) |
| [Fira Code](https://github.com/tonsky/FiraCode) | Google Fonts build | SIL OFL 1.1 | `src/resources/fonts/FiraCode-*.woff2` **and** `docs/fonts/FiraCode-*.woff2` | [`src`](src/resources/fonts/LICENSE.FiraCode), [`docs`](docs/fonts/LICENSE.FiraCode) |

- Mol*: Copyright (c) 2017 - now, Mol* contributors. Vendored 2026-07-15 from
  the official npm package `molstar@5.10.1`, whose version is embedded in the
  JavaScript bundle. After removing only this project's three-line attribution
  banner, both files match the official jsDelivr npm artifacts byte-for-byte:
  `molstar.js` SHA-256 `5567eb19fa8e7a7b3b161d4b96807c4db244cecd2f3e7c87f99c052b8b5b5b30`;
  `molstar.css` SHA-256 `5b68ceb6d3642549b4e9b2c071e58e41b98a5350ae269180587b39da86925d55`.
- ESM: the unmodified `py3-none-any` wheel was built from the upstream commit
  where the MIT relicense landed. Its SHA-256 is
  `d5e412470877fa2e21c36b40a52cdf1bef5664234654355dc2a35bb8cd2f4d82`;
  `src/resources/wheels/manifest.json` records its size and provenance. The
  wheel pins Chan Zuckerberg Biohub's Apache-2.0 Transformers fork at commit
  `3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf`.
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
| `marked.min.js` | `Web_Server.py` static route `/agent_resource/` |
| `fonts.css`, `*.woff2` | `Web_Server.py` static route `/fonts/` |
| KaTeX CSS/JS | `SSN_Tools.py`, via a `file://` baseUrl on `setHtml()` |
| Fonts in `docs/list_of_commands.html` | relative path, page is opened from disk |

Remaining network use is by design rather than asset loading: model weights are
downloaded from Hugging Face on first use and cached, the ESM C 6B backend calls
a remote inference API, and the agent calls whichever LLM endpoint is configured
(the Ollama, LM Studio and llama.cpp options are fully local).

## 3. Python dependencies

Installed at runtime via `pip`/`uv`; except for the ESM wheel identified in
section 1, these packages are not redistributed by this project. As of the
PySide6 and graspologic-native migrations there are **no
strong-copyleft dependencies remaining**; what is left is weak/file-level
copyleft, which imposes obligations only on those packages' own files. This set
is compatible with the project's Apache-2.0 license.

PySide6 is used under its **LGPL-3.0** option and is installed separately; Qt is
not included in this source repository. A normal `pip install` keeps the Qt
libraries separate and replaceable. Any future executable or installer that
redistributes Qt binaries needs a separate LGPL compliance review.

### Direct dependencies

| Package | License |
|---|---|
| PySide6 (Essentials + Addons) | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0 — used under **LGPL-3.0** |
| shiboken6 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0 — used under **LGPL-3.0** |
| graspologic-native | MIT (Copyright (c) Microsoft Corporation) |
| biopython | Biopython License (BSD-style) |
| esm 3.3.0 wheel | MIT — bundled as documented in section 1 |
| torch | BSD-3-Clause — CPU, CUDA, XPU, Linux ROCm, or Windows ROCm build selected at installation |
| AMD ROCm 7.14 runtime wheel components | MIT — downloaded dynamically from AMD's official index only for supported Windows AMD GPUs |
| Biohub Transformers fork, accelerate | Apache-2.0 |
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

`esm` 3.3.0 additionally pulls in the following, all permissive: biotite and
RDKit (BSD-3-Clause), boto3, pygtrie and tenacity (Apache-2.0), pydssp, py3dmol,
dna-features-viewer, einops, msgpack-numpy and cloudpathlib (MIT), plus
ipywidgets (BSD-3-Clause).

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
runtime from Hugging Face or accessed through remote APIs and remain governed by
their publishers' terms. For models with explicit usage restrictions, the
application may require a local acknowledgement before accessing model files;
that acknowledgement is not a sublicense or a substitute for reading the terms.

| Model family | Source | License |
|---|---|---|
| ESM-2 | `facebook/esm2_*` | MIT |
| ESM C 300M | `biohub/esmc-300m-2024-12` | MIT (card also carries a legacy `other` tag) |
| ESM C 600M | `biohub/esmc-600m-2024-12` | MIT (card also carries a legacy `other` tag) |
| ESM3-open 1.4B | `biohub/esm3-sm-open-v1` | MIT |
| ESM C 6B (remote API) | `https://biohub.ai` | Governed by the API provider's terms of use |
| ProtBERT | [`Rostlab/prot_bert`](https://huggingface.co/Rostlab/prot_bert) | Academic Free License 3.0 (AFL-3.0). The Hugging Face repository has no license tag, but the official [`ProtTrans`](https://github.com/agemagician/ProtTrans#license) project explicitly states that its pretrained models are released under AFL-3.0. ProtTrans source code is separately MIT-licensed. |
| ProstT5 | [`Rostlab/ProstT5_fp16`](https://huggingface.co/Rostlab/ProstT5_fp16) | MIT (declared by the model repository) |
| Ankh Base, Ankh Large | [`ElnaggarLab/ankh-base`](https://huggingface.co/ElnaggarLab/ankh-base), [`ankh-large`](https://huggingface.co/ElnaggarLab/ankh-large) | CC-BY-NC-SA-4.0 — non-commercial, attribution, and ShareAlike terms; explicit application acknowledgement required before access |

> **Note on the `esm` package licensing history.** EvolutionaryScale's PyPI
> releases through 3.2.3 carry the bespoke Cambrian license. The upstream MIT
> relicense landed on 2026-05-27 at commit
> `c94ed8d763bbd7088b296949e5b401e8ea12073a`, after those releases. This project
> therefore builds and redistributes an unmodified wheel from that exact
> MIT-licensed commit rather than relabeling a Cambrian-licensed PyPI artifact.
> The adjacent license and recorded hash travel with the wheel.
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
> process** (`src/resources/esmfold/esmfold_worker.py`). Model weights remain
> external and are not included in the bundled wheel.

## 6. Artwork

The project author confirmed on 2026-08-09 that the logos and icons in
`src/bin/logos/` and screenshots in `docs/assets/` are original or otherwise
authorized. This satisfies the artwork-authorship check recorded in
[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md). Authority to release those works
remains subject to the same University of Toronto ownership review as the rest
of the project.

---

## Maintenance

Re-run this review whenever a dependency is added, a bundled asset is refreshed,
or a new model backend is introduced. When refreshing a bundled asset, copy the
upstream `LICENSE` file alongside it, record the version above, and re-apply the
attribution banner.
