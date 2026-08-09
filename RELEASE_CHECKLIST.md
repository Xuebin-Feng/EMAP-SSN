# Release Clearance Checklist

This checklist separates completed technical provenance work from decisions
that require the copyright holder or an authorized institution. Do not publish
a release while any item under **External authority and authorship** is open.

## External authority and authorship

- [ ] Confirm whether the University of Toronto has ownership or approval rights
  and retain the decision reference outside the public repository.
- [ ] Replace the provisional `Copyright 2026 Xuebin Feng` wording everywhere
  with the exact owner wording authorized by that review.
- [x] Confirm that the three commits authored as
  `VS Code <vscode@users.noreply.github.com>` are user-controlled automation,
  or obtain the necessary contribution permission. Confirmed by the project
  author on 2026-08-09.
- [x] Confirm that project-authored source code, prompts, and documentation are
  original or otherwise authorized for release by the approved copyright
  holder. Authorship confirmed by the project author on 2026-08-09; final
  release authority remains subject to the ownership review above.
- [x] Confirm that the logos and icons in `src/bin/logos/` and screenshots in
  `docs/assets/` are original or authorized. Remove, replace, or attribute any
  asset that cannot be confirmed. Authorship confirmed by the project author
  on 2026-08-09; final release authority remains subject to the ownership
  review above.

## Completed provenance checks

- [x] Mol* JavaScript and CSS identified as version 5.10.1. With the project
  attribution banner removed, both bodies match the official npm CDN artifacts
  byte-for-byte. The body hashes are recorded in `THIRD_PARTY_LICENSES.md`.
- [x] Bundled Mol*, Tabulator, marked, KaTeX, Inter, and Fira Code assets map to
  adjacent license texts in `THIRD_PARTY_LICENSES.md`.
- [x] Ankh Base and Large are identified as separately downloaded
  CC-BY-NC-SA-4.0 weights and require explicit acknowledgement before access.
- [x] ProtBERT is distinguished from the unrelated ProteinBERT project. The
  official ProtTrans project licenses its pretrained models under AFL-3.0.
- [x] DirectML has been removed. Supported Windows 11 AMD GPUs install the
  pinned ROCm/PyTorch build from AMD's official index; unsupported AMD systems
  fall back to CPU.
- [x] The bundled ESM 3.3.0 wheel is built from the exact upstream MIT commit
  and mapped to its adjacent license, source commit, size, and SHA-256 manifest.

## Final technical gate

- [ ] Run `python release_checks/release_audit.py` from the repository root.
- [ ] Run `python -m unittest discover -s release_checks -p "test_*.py"`.
- [ ] After committing the intended release contents, run
  `python release_checks/release_audit.py --archive-ref HEAD --strict-release`.
- [ ] Review the resulting source archive and confirm that it contains no model
  weights, embeddings, input data, credentials, caches, or local tool settings.
