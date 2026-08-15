# Bundled Python wheels

This directory contains two wheels used by the launchers:

- The ESM 3.3.0 wheel is unmodified. It was built from the first upstream commit
  that carries the MIT license, because the older ESM release on PyPI carries
  the Cambrian license.
- The Biohub Transformers wheel comes from commit
  `3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf`, whose Hugging Face upstream base
  is `753d61104116eefc8ffc977327b441ee0c8d599f`. The only source change is the
  accompanying packaging patch, which changes `4.57.6` to the distinguishable
  local version `4.57.6+biohub.3a8956f`.

`manifest.json` records both source commits, wheel hashes, sizes, licenses, and
the Transformers base commit. `esm-3.3.0-runtime-requirements.txt` is derived
from every ESM `Requires-Dist` entry except `torch` and `transformers`; a test
checks that it remains synchronized with the verified ESM wheel.

## Rebuilding Biohub Transformers

Clone `https://github.com/Biohub/transformers.git`, check out the full source
commit above, apply `transformers-4.57.6+biohub.3a8956f.patch`, and build from a
clean checkout with:

```powershell
$env:SOURCE_DATE_EPOCH = '1779863841'
uv build --wheel --python C:\Python312\python.exe --out-dir wheelhouse
```

The resulting wheel must have size `12139612` bytes and SHA-256
`74cb19ba0b6c4cf0769322f0ef035bd016eea6ccb2f587a1ff1263a016354c3b`.
The Python path is illustrative; any clean supported Python build frontend may
be used, but the fixed source-date epoch and clean source tree are required for
the recorded reproducible artifact.

PyTorch wheels are intentionally not stored here. `Install_Dependencies.py`
selects a pinned CPU, CUDA, XPU, Linux ROCm, Apple MPS, or Windows ROCm build and
downloads it from the official PyTorch or AMD package index. The package manager
may retain its normal per-user cache, but the project creates no ROCm wheel
cache. Ordinary PyPI dependencies are also downloaded. Consequently, first
installation requires internet access and this is not a fully offline install.

Model weights are also not bundled here. Each model plug-in downloads or accesses
weights separately under the publisher's terms.
