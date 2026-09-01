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
import EMAPSSN_Config as cfg
import web_ui.esmfold_backend as esmfold_backend
from PySide6.QtWidgets import QApplication, QMessageBox
from utilities.Terminal_Launcher import (
    HoldMode,
    TerminalUnavailableError,
    launch_in_terminal,
)

def print_help():
    print("""
    ESM3 3D Structure Prediction
    ============================
    Usage:
      esmfold
          With exactly 1 node selected, folds it using ESM3 1.4B (biohub/esm3-sm-open-v1).
          With no node selected, registers the sidebar button "🧬 Fold View" and opens the Mol* viewer in the browser.
      esmfold multi
          Folds all currently selected nodes sequentially using ESM3 1.4B (biohub/esm3-sm-open-v1).
      esmfold large
          Folds exactly 1 selected node through the Biohub API using the ESM3 model configured in Biohub_API.json.
      esmfold large multi
      esmfold multi large
          Folds all selected nodes sequentially through the configured Biohub ESM3 API model.
      esmfold help
          Displays this help message.
    """)

def sanitize_filename(name):
    import re
    # Replace any character that is not alphanumeric, a dash, dot, or underscore with '_'
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)


def _set_console_text(viewer, message):
    if hasattr(viewer, 'console_text'):
        viewer.console_text.text = message


def _report_usage_error(viewer, message):
    print(f"Error: {message}")
    print("Usage: esmfold [large] [multi]")
    _set_console_text(viewer, f"Error: {message}")


def _parse_options(viewer, args):
    normalized = [str(argument).lower() for argument in args]
    allowed = {"large", "multi"}
    unknown = [argument for argument in normalized if argument not in allowed]
    if unknown:
        _report_usage_error(viewer, f"Unknown esmfold keyword: {unknown[0]}")
        return None
    duplicates = sorted({argument for argument in normalized if normalized.count(argument) > 1})
    if duplicates:
        _report_usage_error(viewer, f"Duplicate esmfold keyword: {duplicates[0]}")
        return None
    return {
        "large": "large" in normalized,
        "multi": "multi" in normalized,
    }


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
        if len(args) != 1:
            _report_usage_error(viewer, "Help cannot be combined with other keywords.")
            return
        print_help()
        _set_console_text(viewer, "Help information printed to the terminal")
        return

    options = _parse_options(viewer, args)
    if options is None:
        return
    is_large = options["large"]
    is_multi = options["multi"]

    # 3. Determine selected nodes
    selected_indices = getattr(viewer, 'selected_indices', [])
    if not selected_indices:
        node_idx = getattr(viewer, 'selected_node_idx', None)
        if node_idx is not None:
            selected_indices = [node_idx]

    if not selected_indices:
        if not args:
            esmfold_backend.register(viewer)
            esmfold_backend.open_esmfold_ui(viewer)
            return

        print("Error: No nodes selected. Please select a node in the visualizer first.")
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Error: No nodes selected."
        return

    # 4. Check for multiple selections vs "multi" command flag
    if len(selected_indices) > 1 and not is_multi:
        print("Error: Multiple nodes selected. Run 'esmfold multi' to fold them, or select a single node.")
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Error: Multiple nodes selected. Use 'esmfold multi'."
        return

    # 5. Select hardware only for local inference. Biohub runs remotely.
    device_str = None
    if not is_large:
        try:
            from utilities import Hardware_Utils
            import torch
        except ImportError:
            print("Error: PyTorch or Hardware_Utils could not be imported.")
            _set_console_text(viewer, "Error: PyTorch/Hardware_Utils missing")
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
    structures_dir = esmfold_backend.get_structures_directory()
    os.makedirs(structures_dir, exist_ok=True)
    
    # Register web button and route mapping
    esmfold_backend.register(viewer)

    try:
        action_url = viewer.get_web_url("/api/action")
    except Exception as error:
        message = f"ESMFold cannot start because the Viewer web server is unavailable:\n{error}"
        print(f"Error: {message}")
        parent = getattr(viewer, 'main_window', None)
        QMessageBox.critical(parent, "ESMFold Web Server Error", message)
        _set_console_text(viewer, "Error: Viewer web server unavailable.")
        return

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

    if is_large:
        print("Launching Biohub ESM3 structure prediction worker in a separate console...")
        cmd = [
            python_exe,
            abs_worker_script,
            tmp_path,
            abs_structures_dir,
            "--mode",
            "large",
            "--action-url",
            action_url,
        ]
        terminal_title = "SSN ESMFold Large"
    else:
        print("Launching local ESM3 3D structure prediction background worker in a separate console...")
        cmd = [
            python_exe,
            abs_worker_script,
            tmp_path,
            abs_structures_dir,
            device_str,
            "--action-url",
            action_url,
        ]
        terminal_title = "SSN ESMFold"
    try:
        launch_in_terminal(
            cmd,
            cwd=project_root,
            hold=HoldMode.NEVER,
            title=terminal_title,
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
    esmfold_backend.open_esmfold_ui(viewer, show_existing_dialog=False)
    mode_label = "Biohub ESM3" if is_large else "local ESM3"
    _set_console_text(
        viewer,
        f"Spawning separate console to fold {len(nodes_to_fold)} structure(s) with {mode_label}...",
    )
