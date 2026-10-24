# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Load the Configuration window (ConfigGUI) without the developer's files.

EMAPSSN_Config defines ConfigGUI inside its ``__main__`` block, and the window
constructor always reads <project>/viewer_settings.json
(ConfigGUI._read_custom_settings ignores the environment) and then hashes the
inputs that file selects. load_config_namespace() runs the module source up to
its application block, so no launcher, IPC, excepthook or event loop starts,
with SSN_VIEWER_SETTINGS_PATH pointed at a missing file, and leaves the test
process's output streams as it found them. open_config_window()
builds a window whose directory settings all point into a given folder.

Not collected by unittest; import with ``from tests.config_gui_loader import ...``.
"""
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
CONFIG_PATH = SRC / "EMAPSSN_Config.py"
# First line of the application block; the launcher and event loop follow it.
APPLICATION_MARKER = "    existing_qt_application = QApplication.instance()"


def load_config_namespace():
    """Return the EMAPSSN_Config ``__main__`` namespace up to the application block."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    if APPLICATION_MARKER not in text:
        # Without it, exec would run on into the event loop and hang.
        raise AssertionError("EMAPSSN_Config.py no longer contains the launcher marker")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    namespace = {"__name__": "__main__", "__file__": str(CONFIG_PATH)}
    excepthook = sys.excepthook
    streams = [
        (stream, stream.encoding, stream.errors)
        for stream in (sys.stdout, sys.stderr)
        if hasattr(stream, "reconfigure")
    ]
    try:
        # SSN_NO_CONSOLE_CAPTURE: the entry point's first step would take over this process's
        # standard streams (utilities.Console_Capture); they are the test runner's.
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"SSN_VIEWER_SETTINGS_PATH": str(Path(directory) / "missing.json"),
                         "SSN_NO_CONSOLE_CAPTURE": "1"}
        ):
            source = text.split(APPLICATION_MARKER)[0]
            exec(compile(source, str(CONFIG_PATH), "exec"), namespace)
    finally:
        # The application block installs _exit_on_uncaught_exception; never keep it.
        sys.excepthook = excepthook
        # Nor the startup's switch to UTF-8 (utilities.Output_Streams): these
        # streams are the test runner's.
        for stream, encoding, errors in streams:
            stream.reconfigure(encoding=encoding, errors=errors)
    return namespace


def directory_settings(root):
    """Custom settings whose base directories (and so every derived one) lie in root.

    EMBED_DIR is a base directory of its own (the project's Embeddings
    folder, under none of the others), so it is set in root too.
    """
    root = Path(root)
    return {
        "INPUT_FILE_DIR": str(root / "inputs"),
        "CACHE_FILE_DIR": str(root / "cache"),
        "ANALYSIS_RESULT_DIR": str(root / "results"),
        "EMBED_DIR": str(root / "Embeddings"),
    }


def directory_globals(namespace, root):
    """The module's directory settings, every one of them in root.

    The base directories are directory_settings(root); each other directory is
    its default under them.
    """
    settings = directory_settings(root)
    resolve = namespace["resolve_directory_path"]
    return {
        key: settings.get(key) or resolve(default, base_directories=settings)
        for key, default in namespace["DIRECTORY_PROFILE_DEFAULTS"].items()
    }


def open_config_window(gui_class, root):
    """Build a ConfigGUI that reads directory_settings(root) instead of viewer_settings.json.

    While it is built, the module's directory settings point into root too:
    the Inputs tab lists FASTA_DIR, MSA_DIR and HDF5_DIR before it reads its
    settings, and those are relative to the working directory, so a test run
    from a checkout would otherwise list the checkout's own Input_Files.
    """
    namespace = gui_class._read_custom_settings.__globals__
    with mock.patch.dict(namespace, directory_globals(namespace, root)), mock.patch.object(
        gui_class, "_read_custom_settings", return_value=directory_settings(root)
    ):
        return gui_class()


def choose_layout(window, choice):
    """Pick "ssn", "umap" or "pacmap" in a Config window's Layout Method dropdown, as a click does."""
    combo = window.cb_layout_method
    index = combo.findData(choice)
    if index < 0:
        raise AssertionError(f"the Layout Method dropdown has no {choice!r}")
    combo.setCurrentIndex(index)


def finish_statistics(window, timeout=60.0):
    """Let the statistics run a Config window started finish, and show its result.

    run_statistics works on a worker thread and the window shows what it finds when the
    thread's signal reaches the event loop, so a test that asks for the report waits
    here, inside any patch the worker's reads must see.
    """
    import time

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while window._statistics_request is not None:
        if time.monotonic() > deadline:
            raise AssertionError("the network statistics did not finish")
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
