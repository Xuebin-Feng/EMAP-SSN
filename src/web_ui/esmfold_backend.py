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
import threading
from urllib.parse import unquote
import EMAPSSN_Config as cfg
from desktop.Desktop_App import translate
from web_ui import residue_layers
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
    registry.register_api_route(
        "esmfold",
        residue_layers.ROUTE_PREFIX,
        lambda request: handle_residue_layers(registry, request),
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
    return viewer.open_page("esmfold", show_existing_dialog=show_existing_dialog)


def handle_residue_layers(registry, request):
    """Answer the Fold View page's request for a structure's per-residue layers.

    /api/esmfold/layers/<structure>.pdb lists the layers offered for the
    structure, and .../<structure>.pdb/<property> is one layer as a
    MolViewSpec annotation (web_ui/residue_layers.py). Both read the
    residue data file beside the structure in the folder the /structures/
    route serves, and name their files through contained_path as that
    route does, so a request can't reach anything it couldn't fetch there.
    The data is shown only when it describes the PDB (residue_layers.py).
    The web server has checked the request's origin already; this answers
    (status, payload).
    """
    from web_ui.Web_Server import contained_path

    directory = registry.static_routes.get("/structures/")
    parts = [unquote(part) for part in request.path[len(residue_layers.ROUTE_PREFIX):].split("/")]
    structure = parts[0]
    sidecar = residue_layers.sidecar_name(structure)
    if (directory is None or sidecar is None or len(parts) not in (1, 2) or not parts[-1]
            or os.path.basename(structure) != structure or "\\" in structure):
        return 404, {"error": "Not a structure layer."}
    structure_path = contained_path(directory, structure)
    sidecar_path = contained_path(directory, sidecar)
    if structure_path is None or sidecar_path is None:
        return 403, {"error": "Forbidden"}
    if not os.path.isfile(structure_path):
        return 404, {"error": "Structure not found."}
    if len(parts) == 1:
        return 200, residue_layers.layer_listing(structure, structure_path, sidecar_path)
    try:
        annotation = residue_layers.layer_annotation(
            residue_layers.read_residue_data(structure, structure_path, sidecar_path), parts[1]
        )
    except residue_layers.ResidueDataError as error:
        return 404, {"error": str(error)}
    if annotation is None:
        return 404, {"error": "This structure has no such layer."}
    return 200, annotation

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
    there is nobody to fold from. The structures that were to be shown while
    the page was not there (show_structures) follow the session it restores."""
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
    _send_structures_to_show(viewer)


def _structure_event(node_id, pdb_filename, **extra):
    """The esmfold_pdb event that makes the page load the structure file ``pdb_filename`` of node ``node_id``."""
    return {
        "type": "esmfold_pdb",
        "node_id": node_id,
        "pdb_url": f"/structures/{pdb_filename}",
        **extra,
    }


def handle_structure_folded(viewer, data):
    """Broadcasts the esmfold_pdb event when a structure has finished folding in the worker process."""
    node_id = data.get("node_id")
    pdb_filename = data.get("pdb_filename")
    if node_id and pdb_filename:
        _tell_command(viewer, "note_structure_folded", _run_id(data.get("run")), node_id)
        viewer.broadcast_event(_structure_event(node_id, pdb_filename))
        print(f"Structure folded for {node_id}. Broadcasted event to browser.")


# What show_structures holds for a page that is not connected yet, and the lock of the threads that fill and empty it.
_SHOWN_LOCK = threading.Lock()


def _structures_to_show(viewer):
    """The structures waiting for the page, {pdb file name: node id}, in the order they were asked for."""
    waiting = getattr(viewer, "_esmfold_structures_to_show", None)
    if not isinstance(waiting, dict):
        waiting = {}
        viewer._esmfold_structures_to_show = waiting
    return waiting


def _send_structures_to_show(viewer):
    """Send the page the structures that wait for it, as events, and forget them."""
    with _SHOWN_LOCK:
        waiting = _structures_to_show(viewer)
        structures = list(waiting.items())
        waiting.clear()
    for pdb_filename, node_id in structures:
        viewer.broadcast_event(_structure_event(node_id, pdb_filename, saved=True))


def page_connected(viewer):
    """Whether the Fold View page is connected to the Viewer's event stream."""
    has_event_client = getattr(getattr(viewer, "web_server", None), "has_event_client", None)
    return bool(callable(has_event_client) and has_event_client("esmfold"))


def show_structures(viewer, structures):
    """Show structures that are saved already in the Fold View page, as a structure just folded is shown.

    ``structures`` are (node id, PDB file name) pairs, the files being in the
    folder the page's /structures/ route serves. Each reaches the page as the
    esmfold_pdb event a worker's report makes (handle_structure_folded),
    marked "saved": the page loads a saved structure only when its scene does
    not have it already, where a structure just folded is loaded again. A
    page that is not connected (it is being opened, or is not open) is sent
    them when it asks for its session, after the one it restores.
    """
    with _SHOWN_LOCK:
        _structures_to_show(viewer).update((pdb_filename, node_id) for node_id, pdb_filename in structures)
    if page_connected(viewer):
        _send_structures_to_show(viewer)

def handle_console_debug_err(viewer, data):
    """Prints debug error logs from the browser console into the Python terminal."""
    pass
