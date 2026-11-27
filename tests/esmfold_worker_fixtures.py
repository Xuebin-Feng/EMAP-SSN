# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python stand-ins for the Biohub/ESM SDK used by ESMFold worker and layer tests.

Kept free of Qt and Qt WebEngine imports so ``tests/test_esmfold_worker.py``
stays pure Python and decoupled from the Fold View page harness.

Not collected by unittest; import with ``from tests.esmfold_worker_fixtures import ...``.
"""

import contextvars
import sys
import threading
import types
from unittest import mock


class FakeProtein:
    def __init__(self, sequence=None):
        self.sequence = sequence


class FakeProteinError:
    def __init__(self, error_code, error_msg="rejected"):
        self.error_code = error_code
        self.error_msg = error_msg

    def __str__(self):
        return self.error_msg


class FakeGenerationConfig:
    def __init__(self, track, num_steps):
        self.track = track
        self.num_steps = num_steps


class FakeOutput:
    def __init__(self, pdb="ATOM\n", plddt=None):
        self.plddt = plddt
        self.pdb = pdb
        self.plddt_at_write = None

    def to_pdb_string(self):
        self.plddt_at_write = self.plddt
        return self.pdb


class FakeFoldClient:
    """Stands in for the Biohub ESMFold 2 client: fold(sequence) answers as ``answer(sequence)`` says.

    The answer is an output (FakeOutput) or a FakeProteinError. It records the calls, and whether the SDK's own
    retry was switched off in the thread that made each one.
    """

    def __init__(self, answer=None):
        self.answer = answer or (lambda sequence: FakeOutput(pdb=f"PDB {sequence}\n"))
        self.calls = []
        self.retries_skipped = []
        self.closed = False
        self.token = "old-token"
        self.headers = {"Authorization": "Bearer old-token"}
        self._lock = threading.Lock()

    def fold(self, sequence):
        with self._lock:
            self.calls.append(sequence)
            self.retries_skipped.append(sys.modules["esm.sdk.retry"].skip_retries_var.get())
        return self.answer(sequence)

    def close(self):
        self.closed = True


# The saved Biohub settings with their defaults filled in, as Biohub_API.load_api_settings returns them.
REMOTE_SETTINGS = {
    "ESM_API_TOKEN": "hidden", "ESM_API_URL": "https://biohub.ai",
    "FOLD_MODEL": "esmfold2-2026-05", "FOLD_PARALLEL_REQUESTS": 1,
}


def fake_esm_modules():
    esm_module = types.ModuleType("esm")
    sdk_module = types.ModuleType("esm.sdk")
    api_module = types.ModuleType("esm.sdk.api")
    api_module.ESMProtein = FakeProtein
    api_module.ESMProteinError = FakeProteinError
    api_module.GenerationConfig = FakeGenerationConfig
    retry_module = types.ModuleType("esm.sdk.retry")
    retry_module.skip_retries_var = contextvars.ContextVar("skip_retries", default=False)
    forge_module = types.ModuleType("esm.sdk.forge")
    forge_module.SequenceStructureForgeInferenceClient = mock.Mock()
    esm_module.sdk = sdk_module
    sdk_module.api = api_module
    sdk_module.retry = retry_module
    sdk_module.forge = forge_module
    return {
        "esm": esm_module,
        "esm.sdk": sdk_module,
        "esm.sdk.api": api_module,
        "esm.sdk.retry": retry_module,
        "esm.sdk.forge": forge_module,
    }
