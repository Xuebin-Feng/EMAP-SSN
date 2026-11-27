# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the Install_Dependencies test modules.

Not collected by unittest: it defines no TestCase.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Detect_GPU  # noqa: E402
import Install_Dependencies  # noqa: E402


TORCH_DEVIATION_LINE = (
    "The package `esm` requires `torch>=2.11.0,<2.12.0`, "
    f"but `{Install_Dependencies.TORCH_VERSION}+cu132` is installed"
)
TRANSFORMERS_DEVIATION_LINE = (
    "The package `esm` requires `transformers>=4.57.6,<5.0.0`, "
    f"but `{Install_Dependencies.TRANSFORMERS_VERSION}` is installed"
)


def saved_state(active, requirements):
    """The backend state an earlier install saved with `active` validated.

    write_state adds the detection report, hardware fingerprint and requested
    candidate ladder when the test passes it a report.
    """
    return {
        "schema": Install_Dependencies.STATE_SCHEMA,
        "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION,
        "active_backend": Install_Dependencies._spec_payloads((active,))[0],
        "requirements_sha256": Install_Dependencies._requirements_sha256(requirements),
        "esm_version": Install_Dependencies.ESM_VERSION,
        "transformers_version": Install_Dependencies.TRANSFORMERS_VERSION,
        "esm_runtime_requirements_sha256": Install_Dependencies._requirements_sha256(
            requirements.parent / "esm_runtime_requirements.txt"
        ),
        "validated_devices": [],
        "ignored_devices": [],
        "attempts": [],
    }
