# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""What `probe.py install` copies into the checkout's .venv/site-packages, as emapssn_desktop_probe_stub.py.

The .pth file beside it names the checkout and imports this module only when the gate file exists:

    import os; os.path.exists(r'<checkout>/temp/desktop_probe/enabled') and __import__('emapssn_desktop_probe_stub').load(r'<checkout>')

So a venv with the probe installed and the gate off pays one stat call per interpreter start and imports
nothing. With the gate on, ``load`` runs boot.py from the checkout (so the probe's code is the checkout's own and
a change to it needs no reinstall) and lets it decide whether this process is one of the checkout's entry points.
Python 3.13 runs a .pth line twice in a venv, hence the guard. Standard library only, nothing but ``load`` may
raise into the interpreter's start-up, and it does not.
"""

import importlib.util
import os
import sys

_loaded = False


def load(checkout):
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        path = os.path.join(checkout, "tests", "desktop_probe", "boot.py")
        spec = importlib.util.spec_from_file_location("emapssn_desktop_probe_boot", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["emapssn_desktop_probe_boot"] = module
        spec.loader.exec_module(module)
        module.start(checkout)
    except Exception as error:  # the interpreter starts as if the probe were not there
        try:
            directory = os.path.join(checkout, "temp", "desktop_probe")
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "boot-errors.log"), "a", encoding="utf-8") as handle:
                handle.write(f"stub: {type(error).__name__}: {error}\n")
        except Exception:
            pass
