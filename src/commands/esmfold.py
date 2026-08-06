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
from PySide6.QtWidgets import QApplication

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
    import subprocess

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
    
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NEW_CONSOLE
        cmd = [python_exe, abs_worker_script, tmp_path, abs_structures_dir, device_str]
        subprocess.Popen(cmd, creationflags=creation_flags, cwd=project_root)
    elif sys.platform == "darwin":
        # macOS: AppleScript to activate Terminal.app and execute the script in a new window/tab.
        escaped_project_root = project_root.replace('"', '\\"')
        escaped_python = python_exe.replace('"', '\\"')
        escaped_worker = abs_worker_script.replace('"', '\\"')
        escaped_tmp = tmp_path.replace('"', '\\"')
        escaped_structs = abs_structures_dir.replace('"', '\\"')
        escaped_device = device_str.replace('"', '\\"')
        
        cmd_str = f'cd "{escaped_project_root}" && "{escaped_python}" "{escaped_worker}" "{escaped_tmp}" "{escaped_structs}" "{escaped_device}"'
        escaped_cmd = cmd_str.replace('"', '\\"')
        
        subprocess.Popen([
            "osascript",
            "-e", 'tell application "Terminal"',
            "-e", 'activate',
            "-e", f'do script "{escaped_cmd}"',
            "-e", 'end tell'
        ])
    else:
        # Linux: Detect available terminal emulator and run command
        import shutil
        terminals = [
            "gnome-terminal", "konsole", "xfce4-terminal", 
            "mate-terminal", "lxterminal", "kitty", 
            "alacritty", "xterm", "x-terminal-emulator"
        ]
        chosen_terminal = None
        for term in terminals:
            if shutil.which(term):
                chosen_terminal = term
                break
        
        escaped_project_root = project_root.replace('"', '\\"')
        escaped_python = python_exe.replace('"', '\\"')
        escaped_worker = abs_worker_script.replace('"', '\\"')
        escaped_tmp = tmp_path.replace('"', '\\"')
        escaped_structs = abs_structures_dir.replace('"', '\\"')
        escaped_device = device_str.replace('"', '\\"')
        
        cmd_str = f'cd "{escaped_project_root}" && "{escaped_python}" "{escaped_worker}" "{escaped_tmp}" "{escaped_structs}" "{escaped_device}"'
        
        if chosen_terminal:
            if chosen_terminal in ["gnome-terminal", "kitty", "alacritty"]:
                subprocess.Popen([chosen_terminal, "--", "bash", "-c", cmd_str])
            elif chosen_terminal == "konsole":
                subprocess.Popen(["konsole", "-e", "bash", "-c", cmd_str])
            else:
                subprocess.Popen([chosen_terminal, "-e", f"bash -c '{cmd_str}'"])
        else:
            # Fallback to background process if no terminal emulator is found
            cmd = [python_exe, abs_worker_script, tmp_path, abs_structures_dir, device_str]
            subprocess.Popen(cmd, cwd=project_root)
            try:
                from PySide6.QtWidgets import QMessageBox
                parent = getattr(viewer, 'main_window', None)
                QMessageBox.warning(
                    parent, "No Terminal Emulator Found",
                    "Could not locate a terminal emulator (e.g. gnome-terminal, xterm). "
                    "The structure prediction script has been launched in the background, but console progress output will not be visible."
                )
            except Exception:
                print("Warning: Could not locate a terminal emulator. Script running in background.")

    # 10. Open Mol* web browser tab immediately
    esmfold_backend.open_esmfold_ui(viewer)
    if hasattr(viewer, 'console_text'):
        viewer.console_text.text = f"Spawning separate console to fold {len(nodes_to_fold)} structure(s)..."
