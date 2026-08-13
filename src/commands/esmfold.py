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
import sys
import Command_Engine
import SSN_Config as cfg
import SSN_Utils as utils
import web_ui.esmfold_backend as esmfold_backend
from PySide6.QtWidgets import QApplication, QMessageBox
from utilities.Terminal_Launcher import (
    HoldMode,
    TerminalUnavailableError,
    launch_in_terminal,
)

def print_help():
    print("""
    Local ESM3 3D Structure Prediction
    ==================================
    Usage:
      esmfold
          Folds the currently selected node (only if exactly 1 node is selected) using ESM3 1.4B (biohub/esm3-sm-open-v1).
          Registers the sidebar button "🧬 Fold View" and opens the Mol* viewer in the browser.
      esmfold multi
          Folds all currently selected nodes sequentially using ESM3 1.4B (biohub/esm3-sm-open-v1).
      esmfold help
          Displays this help message.
    """)

def sanitize_filename(name):
    import re
    # Replace any character that is not alphanumeric, a dash, dot, or underscore with '_'
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)

def run(viewer, args):
    import warnings
    # Suppress library-level user warnings from esm library
    warnings.filterwarnings("ignore", category=UserWarning, module="esm")

    # 1. Registration callback support
    if args and args[0] == '--register-only':
        esmfold_backend.register(viewer)
        return

    # 2. Help & Usage Check
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    # 3. Determine selected nodes
    selected_indices = getattr(viewer, 'selected_indices', [])
    if not selected_indices:
        node_idx = getattr(viewer, 'selected_node_idx', None)
        if node_idx is not None:
            selected_indices = [node_idx]

    if not selected_indices:
        print("Error: No nodes selected. Please select a node in the visualizer first.")
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Error: No nodes selected."
        return

    # 4. Check for multiple selections vs "multi" command flag
    is_multi = len(args) >= 1 and args[0].lower() == 'multi'
    if len(selected_indices) > 1 and not is_multi:
        print("Error: Multiple nodes selected. Run 'esmfold multi' to fold them, or select a single node.")
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Error: Multiple nodes selected. Use 'esmfold multi'."
        return

    # 5. Check Hardware & VRAM via Hardware_Utils
    try:
        from utilities import Hardware_Utils
        import torch
    except ImportError:
        print("Error: PyTorch or Hardware_Utils could not be imported.")
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Error: PyTorch/Hardware_Utils missing"
        return

    device = Hardware_Utils.get_optimal_device()
    device_str = str(device)
    print(f"Optimal device selected: {device_str}")
    if device.type == 'cpu':
        print("Warning: Running local ESM3 on CPU will be extremely slow.")

    # 6. Parse sequences from FASTA subset/main database
    if not hasattr(viewer, 'sequences_map'):
        viewer.sequences_map = {}
        fasta_path = getattr(cfg, 'NODE_FASTA_FILE', None) or getattr(cfg, 'SEQUENCES_FILE', '')
        if fasta_path and os.path.exists(fasta_path):
            try:
                from Bio import SeqIO
                for rec in SeqIO.parse(fasta_path, "fasta"):
                    viewer.sequences_map[rec.id] = str(rec.seq)
                    viewer.sequences_map[rec.description] = str(rec.seq)
            except Exception as e:
                print(f"Warning: Failed to parse FASTA for sequences: {e}")

    # Resolve target sequences to fold
    nodes_to_fold = []
    for idx in selected_indices:
        full_header = viewer.full_headers[idx]
        rec_id = full_header.split()[0]
        
        sequence = None
        if full_header in viewer.sequences_map:
            sequence = viewer.sequences_map[full_header]
        elif rec_id in viewer.sequences_map:
            sequence = viewer.sequences_map[rec_id]
            
        if sequence:
            nodes_to_fold.append((rec_id, sequence))
        else:
            print(f"Warning: Sequence not found in FASTA for node: {rec_id}")

    if not nodes_to_fold:
        print("Error: Could not retrieve sequences for selected nodes.")
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Error: Sequence retrieval failed."
        return

    # 8. Set up Directory & Web Registration
    structures_dir = getattr(cfg, 'STRUCTURES_DIR', os.path.join("Cache_Files", "Structures"))
    os.makedirs(structures_dir, exist_ok=True)
    
    # Register web button and route mapping
    esmfold_backend.register(viewer)

    # 9. Save nodes to fold to a temporary JSON file and spawn background worker process
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as tmp:
        json.dump(nodes_to_fold, tmp, indent=2)
        tmp_path = tmp.name

    python_exe = sys.executable
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(src_dir)
    worker_script = os.path.join(src_dir, "resources", "esmfold", "esmfold_worker.py")

    abs_structures_dir = os.path.abspath(structures_dir)
    abs_worker_script = os.path.abspath(worker_script)

    print("Launching local ESM3 3D structure prediction background worker in a separate console...")
    
    device_str = str(device)
    cmd = [python_exe, abs_worker_script, tmp_path, abs_structures_dir, device_str]
    try:
        launch_in_terminal(
            cmd,
            cwd=project_root,
            hold=HoldMode.NEVER,
            title="SSN ESMFold",
        )
    except (OSError, TerminalUnavailableError) as error:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        message = f"Could not launch the ESMFold worker in a terminal:\n{error}"
        print(f"Error: {message}")
        parent = getattr(viewer, 'main_window', None)
        QMessageBox.critical(parent, "ESMFold Launch Error", message)
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Error: Could not launch the ESMFold terminal."
        return

    # 10. Open Mol* web browser tab immediately
    esmfold_backend.open_esmfold_ui(viewer)
    if hasattr(viewer, 'console_text'):
        viewer.console_text.text = f"Spawning separate console to fold {len(nodes_to_fold)} structure(s)..."
