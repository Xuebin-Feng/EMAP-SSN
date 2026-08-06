# Copyright 2026 Xuebin Feng
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
import SSN_Utils as utils
import SSN_Config as cfg
import webbrowser

def register(viewer):
    """
    Registers custom action handlers and maps static routes for the local ESMFold
    structure prediction and Mol* visualization tool.
    """
    if not hasattr(viewer, "web_action_handlers"):
        viewer.web_action_handlers = {}
        
    viewer.web_action_handlers["save_molstar_session"] = lambda data: handle_save_session(viewer, data)
    viewer.web_action_handlers["load_molstar_session"] = lambda data: handle_load_session(viewer, data)
    viewer.web_action_handlers["structure_folded"] = lambda data: handle_structure_folded(viewer, data)
    viewer.web_action_handlers["console_debug_err"] = lambda data: handle_console_debug_err(viewer, data)
    
    # Register the static route for structures in the Web Server
    if hasattr(viewer, "web_server") and viewer.web_server:
        structures_dir = getattr(cfg, 'STRUCTURES_DIR', os.path.join("Cache_Files", "Structures"))
        os.makedirs(structures_dir, exist_ok=True)
        viewer.web_server.static_routes["/structures/"] = structures_dir
        
        # Register the static route for esmfold assets
        _SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        esmfold_resources_dir = os.path.join(_SRC_DIR, "resources", "esmfold")
        viewer.web_server.static_routes["/esmfold/"] = esmfold_resources_dir
        
    # Register the sidebar button
    if hasattr(viewer, 'add_sidebar_button'):
        viewer.add_sidebar_button(
            "fold_view_btn",
            "🧬 Fold View",
            lambda: open_esmfold_ui(viewer, force=True),
            "Open ESMFold & Mol* structure viewer"
        )

def open_esmfold_ui(viewer, force=False):
    """Opens the local Mol* page in the user's default browser."""
    # Check if there is already an active EventSource connection queue
    is_already_connected = False
    if hasattr(viewer, 'web_server') and viewer.web_server:
        with viewer.web_server.queues_lock:
            is_already_connected = len(viewer.web_server.event_queues) > 0
            
    if not force and is_already_connected:
        return
    webbrowser.open("http://localhost:8000/esmfold.html")
    if hasattr(viewer, 'console_text'):
        viewer.console_text.text = "ESMFold Mol* UI opened in browser"

def handle_save_session(viewer, data):
    """Saves the serialized Mol* JSON session snapshot to the active layout cache folder."""
    try:
        session_data = data.get("session")
        if session_data is None:
            return
            
        cache_path, _ = utils.get_cache_filename()
        layout_dir = os.path.dirname(cache_path)
        os.makedirs(layout_dir, exist_ok=True)
        
        session_file = os.path.join(layout_dir, "molstar_session.json")
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
    except Exception as e:
        print(f"Error saving Mol* session: {e}")

def handle_load_session(viewer, data):
    """Loads the Mol* JSON session snapshot from the active layout cache folder and broadcasts it."""
    try:
        cache_path, _ = utils.get_cache_filename()
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
