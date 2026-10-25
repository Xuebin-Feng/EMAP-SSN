# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import re
import EMAPSSN_Config as cfg
from desktop.Desktop_App import translate
from utilities.Localization import Message
from web_ui.Plugin_Manager import ensure_registry
from desktop.Viewer_State import resolve_selected_cache

STRUCTURES_DIRECTORY = os.path.join("$cache_file$", "Predicted_Structures")


def get_structures_directory():
    """Return the command and web route's shared effective output directory."""
    return cfg.resolve_directory_path(STRUCTURES_DIRECTORY)


def register_backend(registry, viewer):
    """Register Mol* web capabilities without adding a Console button."""
    registry.register_action(
        "esmfold",
        "save_molstar_session",
        lambda data: handle_save_session(viewer, data),
    )
    registry.register_action(
        "esmfold",
        "load_molstar_session",
        lambda data: handle_load_session(viewer, data),
    )
    registry.register_action(
        "esmfold",
        "structure_folded",
        lambda data: handle_structure_folded(viewer, data),
    )
    registry.register_action(
        "esmfold",
        "esmfold_job",
        lambda data: handle_esmfold_job(viewer, data),
    )
    registry.register_action(
        "esmfold",
        "esmfold_open_log",
        lambda data: handle_open_log(viewer, data),
    )
    registry.register_action(
        "esmfold",
        "console_debug_err",
        lambda data: handle_console_debug_err(viewer, data),
    )
    structures_dir = get_structures_directory()
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    registry.register_static_route("esmfold", "/structures/", structures_dir)
    registry.register_static_route(
        "esmfold", "/esmfold/", os.path.join(src_dir, "resources", "esmfold")
    )


def activate(viewer):
    """Add the Fold View button to the Console's button row without creating output directories."""
    if hasattr(viewer, 'add_console_button'):
        viewer.add_console_button(
            "fold_view_btn",
            translate("Viewer", "Fold View"),
            lambda: open_esmfold_ui(viewer),
            translate("Viewer", "Open ESMFold & Mol* structure viewer"),
            icon="dna",
        )


def register(viewer):
    """Compatibility wrapper: ensure backend registration, then activate its UI."""
    registry = ensure_registry(viewer)
    register_backend(registry, viewer)
    registry.registered_plugins.add("esmfold")
    return activate(viewer)

def open_esmfold_ui(viewer, *, show_existing_dialog=True):
    """Opens the local Mol* page in the user's default browser."""
    return viewer._open_web_ui(
        "/esmfold.html",
        Message("ESMFold Mol* UI"),
        "esmfold",
        show_existing_dialog=show_existing_dialog,
    )

def handle_save_session(viewer, data):
    """Saves the serialized Mol* JSON session snapshot to the active layout cache folder."""
    try:
        session_data = data.get("session")
        if session_data is None:
            return
            
        cache_path = resolve_selected_cache(cfg)
        layout_dir = os.path.dirname(cache_path)
        os.makedirs(layout_dir, exist_ok=True)
        
        session_file = os.path.join(layout_dir, "molstar_session.json")
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
    except Exception as e:
        print(f"Error saving Mol* session: {e}")

JOB_EVENTS = ("started", "downloading", "progress", "structure_failed", "authentication", "failed", "finished")


def _short_text(value, limit):
    return value[:limit] if isinstance(value, str) else None


def _run_id(value):
    """The id of a run the Viewer started, as the worker echoes it (--run-id), or None for anything else."""
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,32}", value) else None


def _tell_command(viewer, name, *arguments):
    """Tell the esmfold command what the worker reported (the command keeps the run's job), whatever it does with it."""
    try:
        from commands import esmfold
        getattr(esmfold, name)(viewer, *arguments)
    except Exception as error:  # The page is told all the same.
        print(f"Warning: The esmfold command could not take the worker's report: {error}")


def handle_esmfold_job(viewer, data):
    """Broadcasts how a folding run is going (the worker posts it) for the Fold View page to show.

    Only the fields the page shows are passed on, bounded: the worker is a
    local process, but the page is not told what it did not ask for.
    """
    event = data.get("event")
    if event not in JOB_EVENTS:
        return
    message = {"type": "esmfold_job", "event": event}
    for key in ("total", "index", "succeeded", "percent", "status"):
        value = data.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 10 ** 9:
            message[key] = value
    for key, limit in (("name", 200), ("reason", 500), ("log", 500)):
        text = _short_text(data.get(key), limit)
        if text:
            message[key] = text
    names = data.get("names")
    if isinstance(names, list):
        message["names"] = [_short_text(name, 200) or "" for name in names[:50]]
    if isinstance(data.get("failed"), bool):
        message["failed"] = data["failed"]
    run = _run_id(data.get("run"))
    if run:
        message["run"] = run
        # The log the Viewer made for this run, whatever the worker says its own is.
        log = _log_of(run)
        if log:
            message["log"] = log[:500]
    viewer.broadcast_event(message)
    _tell_command(viewer, "note_job_event", message)


def _log_of(run_id):
    try:
        from commands import esmfold
        return esmfold.log_of(run_id)
    except Exception as error:
        print(f"Warning: The esmfold command could not name the log of a run: {error}")
        return None


def handle_open_log(viewer, data):
    """The page's Open log button: open the log of the run it names in the system's text viewer.

    The run is found by its id among those this Viewer started; a path in the request is never read.
    Where no one is at the Viewer, or the log is gone, nothing opens and the page is told.
    """
    from desktop.Viewer_Jobs import open_log_file
    run = _run_id(data.get("run"))
    if not open_log_file(viewer, _log_of(run) if run else None):
        viewer.broadcast_event({"type": "esmfold_job", "event": "log_not_opened"})


def handle_load_session(viewer, data):
    """Loads the Mol* JSON session snapshot from the active layout cache folder and broadcasts it.

    The page asks when it connects, so the Viewer also says then whether it has
    a window: a page opened by its address in a Viewer with none explains why
    there is nobody to fold from."""
    shows_windows = getattr(viewer, "shows_windows", None)
    if callable(shows_windows):
        viewer.broadcast_event({"type": "viewer_info", "hidden": not shows_windows()})
    try:
        cache_path = resolve_selected_cache(cfg)
        layout_dir = os.path.dirname(cache_path)
        session_file = os.path.join(layout_dir, "molstar_session.json")

        if os.path.exists(session_file):
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
            viewer.broadcast_event({"type": "restore_session", "session": session_data})
        else:
            viewer.broadcast_event({"type": "restore_session", "session": None})
    except Exception as e:
        print(f"Error loading Mol* session: {e}")
        viewer.broadcast_event({"type": "restore_session", "session": None})

def handle_structure_folded(viewer, data):
    """Broadcasts the esmfold_pdb event when a structure has finished folding in the worker process."""
    node_id = data.get("node_id")
    pdb_filename = data.get("pdb_filename")
    if node_id and pdb_filename:
        _tell_command(viewer, "note_structure_folded", _run_id(data.get("run")), node_id)
        pdb_url = f"/structures/{pdb_filename}"
        viewer.broadcast_event({
            "type": "esmfold_pdb",
            "node_id": node_id,
            "pdb_url": pdb_url
        })
        print(f"Structure folded for {node_id}. Broadcasted event to browser.")

def handle_console_debug_err(viewer, data):
    """Prints debug error logs from the browser console into the Python terminal."""
    pass
