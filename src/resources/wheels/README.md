# Bundled Python wheel

This directory contains the unmodified ESM wheel used by the launchers. It was
built from the first upstream commit that carries the MIT license, because the
older ESM release on PyPI carries the Cambrian license. `manifest.json` records
the source commit, wheel hash, size, and license mapping; the release audit
verifies those values.

PyTorch wheels are intentionally not stored here. `Install_Dependencies.py`
selects a pinned CPU, CUDA, XPU, Linux ROCm, Apple MPS, or Windows ROCm build and
downloads it from the official PyTorch or AMD package index. The package manager
may retain its normal per-user cache, but the project creates no ROCm wheel
cache. Consequently, first installation requires internet access and is not an
offline-reproducibility mechanism.

Model weights are also not bundled here. Each model plug-in downloads or accesses
weights separately under the publisher's terms.
