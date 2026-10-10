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
import sys
import Command_Engine
import EMAPSSN_Config as cfg
import web_ui.esmfold_backend as esmfold_backend
from PySide6.QtWidgets import QApplication, QMessageBox
from desktop.Desktop_App import translate
from utilities.Localization import JoinedMessage, Message
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
          With no node selected, folds the node last clicked and says so. With no node
          clicked either, registers the sidebar button "Fold View" and opens the Mol* viewer in the browser.
      esmfold multi
          Folds all currently selected nodes sequentially using ESM3 1.4B (biohub/esm3-sm-open-v1).
      esmfold large
          Folds exactly 1 selected node through the Biohub API using the ESM3 model configured in Biohub_API.json.
      esmfold large multi
      esmfold multi large
          Folds all selected nodes sequentially through the configured Biohub ESM3 API model.
      esmfold help
          Displays this help message.

    Only one folding worker runs at a time: while one is running, esmfold refuses to
    start another. Wait for it to finish, or close its console window.
    """)

# A command's outcome names the selected nodes it skipped when there are this few.
MAX_LISTED_SKIPPED = 5


def sanitize_filename(name):
    import re
    # Replace any character that is not alphanumeric, a dash, dot, or underscore with '_'
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)


def _set_console_text(viewer, message):
    if hasattr(viewer, 'console_text'):
        Command_Engine.show_status(viewer, message)


def _report_usage_error(viewer, message):
    """Report message, a Message, in English to the terminal and translated on the console line."""
    print(f"Error: {message}")
    Command_Engine.command_failed(viewer, f'Error: {message}')
    print("Usage: esmfold [large] [multi]")
    _set_console_text(viewer, Message("Error: {error}", error=message))


def _open_structure_viewer(viewer):
    """Open the Mol* page; return whether it opened.

    A command run for the MCP or the agent page cannot wait for someone to
    close the dialog that says the page is already open.
    """
    from Viewer_Command_Portal import CURRENT
    if CURRENT.get() is None:
        return esmfold_backend.open_esmfold_ui(viewer)
    return esmfold_backend.open_esmfold_ui(viewer, show_existing_dialog=False)


def _structure_viewer_connected(viewer):
    has_event_client = getattr(getattr(viewer, "web_server", None), "has_event_client", None)
    return bool(callable(has_event_client) and has_event_client("esmfold"))


def _discard_worker_input(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _worker_running(viewer):
    """Whether the folding worker this Viewer started is still running.

    A run for the MCP or the agent page has a WorkerTracker, which polls the
    worker until it reports a final result or exits, so it answers: a terminal
    left open after a failure is not a running worker. A typed run has none, so
    the terminal process that hosts the worker answers instead, which lasts as
    long as the worker does. (Some Linux terminals hand their window to another
    process and exit at once; only a tracker can tell for those.)
    """
    worker = getattr(viewer, "esmfold_worker", None)
    if worker is None:
        return False
    process, tracker = worker
    if tracker is not None:
        try:
            return bool(tracker.timer.isActive())
        except RuntimeError:
            pass  # The tracker's Qt object is already deleted.
    return process is not None and process.poll() is None


def _parse_options(viewer, args):
    normalized = [str(argument).lower() for argument in args]
    allowed = {"large", "multi"}
    unknown = [argument for argument in normalized if argument not in allowed]
    if unknown:
        _report_usage_error(viewer, Message("Unknown esmfold keyword: {keyword}", keyword=unknown[0]))
        return None
    duplicates = sorted({argument for argument in normalized if normalized.count(argument) > 1})
    if duplicates:
        _report_usage_error(viewer, Message("Duplicate esmfold keyword: {keyword}", keyword=duplicates[0]))
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
        Command_Engine.command_succeeded(viewer, 'Registered the structure viewer interface.')
        return

    # 2. Help & Usage Check
    if args and args[0].lower() in ['help', '-h', '--help']:
        if len(args) != 1:
            _report_usage_error(viewer, Message("Help cannot be combined with other keywords."))
            return
        print_help()
        _set_console_text(viewer, Message("Help information printed to the terminal"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    options = _parse_options(viewer, args)
    if options is None:
        return
    is_large = options["large"]
    is_multi = options["multi"]

    # 3. Determine selected nodes. With nothing selected, the node last clicked
    # (the left-click focus) is folded, and the outcome says so.
    selected_indices = getattr(viewer, 'selected_indices', [])
    clicked_note = None
    if not selected_indices:
        node_idx = getattr(viewer, 'selected_node_idx', None)
        if node_idx is not None:
            selected_indices = [node_idx]
            clicked_note = Message(
                "Nothing is selected, so the last clicked node is folded: {node}.",
                node=viewer.full_headers[node_idx].split()[0],
            )

    if not selected_indices:
        if not args:
            esmfold_backend.register(viewer)
            if _open_structure_viewer(viewer):
                Command_Engine.command_succeeded(viewer, "Opened the structure viewer; no folding job was requested.")
            elif _structure_viewer_connected(viewer):
                Command_Engine.command_succeeded(viewer, "The structure viewer is already open; no folding job was requested.")
            else:
                Command_Engine.command_failed(viewer, "The structure viewer was not opened; the Viewer's console line says why.")
            return

        print("Error: No nodes selected. Please select a node in the visualizer first.")
        Command_Engine.command_failed(viewer, 'Error: No nodes selected. Please select a node in the visualizer first.')
        if hasattr(viewer, 'console_text'):
            failure = Message("Error: No nodes selected.")
            Command_Engine.show_status(viewer, failure)
        return

    # 4. Check for multiple selections vs "multi" command flag
    if len(selected_indices) > 1 and not is_multi:
        print("Error: Multiple nodes selected. Run 'esmfold multi' to fold them, or select a single node.")
        Command_Engine.command_failed(viewer, "Error: Multiple nodes selected. Run 'esmfold multi' to fold them, or select a single node.")
        if hasattr(viewer, 'console_text'):
            failure = Message("Error: Multiple nodes selected. Use 'esmfold multi'.")
            Command_Engine.show_status(viewer, failure)
        return

    # 4b. One worker at a time: a second would load a second large model into
    # GPU memory, so it is refused before any hardware or sequence work.
    if _worker_running(viewer):
        failure = Message(
            "An esmfold worker is still running. Wait for it to finish, or close its "
            "console window, before folding again."
        )
        print(f"Error: {failure}")
        Command_Engine.command_failed(viewer, f'Error: {failure}')
        _set_console_text(viewer, Message("Error: {error}", error=failure))
        return
    if clicked_note is not None:
        print(clicked_note)

    # 5. Select hardware only for local inference. Biohub runs remotely.
    device_str = None
    if not is_large:
        try:
            from utilities import Hardware_Acceleration as Hardware_Utils
            import torch
        except ImportError:
            print("Error: PyTorch or Hardware_Utils could not be imported.")
            Command_Engine.command_failed(viewer, 'Error: PyTorch or Hardware_Utils could not be imported.')
            _set_console_text(viewer, Message("Error: PyTorch/Hardware_Utils missing"))
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
    skipped_ids = []
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
            skipped_ids.append(rec_id)

    if not nodes_to_fold:
        print("Error: Could not retrieve sequences for selected nodes.")
        Command_Engine.command_failed(viewer, 'Error: Could not retrieve sequences for selected nodes.')
        if hasattr(viewer, 'console_text'):
            failure = Message("Error: Sequence retrieval failed.")
            Command_Engine.show_status(viewer, failure)
        return

    # 8. Set up Directory & Web Registration
    structures_dir = esmfold_backend.get_structures_directory()
    os.makedirs(structures_dir, exist_ok=True)
    
    # Register web button and route mapping
    esmfold_backend.register(viewer)

    try:
        action_url = viewer.get_web_url("/api/action")
    except Exception as error:
        message = Message("ESMFold cannot start because the Viewer web server is unavailable:\n{error}", error=error)
        print(f"Error: {message}")
        Command_Engine.command_failed(viewer, f'Error: {message}')
        parent = getattr(viewer, 'main_window', None)
        from Viewer_Command_Portal import CURRENT
        if CURRENT.get() is None:
            QMessageBox.critical(parent, translate("Viewer", "ESMFold Web Server Error"), message.display())
        _set_console_text(viewer, Message("Error: Viewer web server unavailable."))
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
            "--delete-input",
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
            "--delete-input",
            "--action-url",
            action_url,
        ]
        terminal_title = "SSN ESMFold"
    from Viewer_Command_Portal import CURRENT
    context = CURRENT.get()
    tracker = None
    try:
        if context is not None:
            from Viewer_Worker_Tracking import WorkerTracker
            tracker = WorkerTracker(context)
            cmd += ['--portal-status', str(tracker.path)]
            if os.environ.get('SSN_VIEWER_HEADLESS') == '1' or os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
                cmd += ['--noninteractive']
        process = launch_in_terminal(
            cmd,
            cwd=project_root,
            hold=HoldMode.NEVER,
            title=terminal_title,
        )
    except (OSError, TerminalUnavailableError) as error:
        if tracker is not None:
            tracker.fail(str(error))
        _discard_worker_input(tmp_path)
        message = Message("Could not launch the ESMFold worker in a terminal:\n{error}", error=error)
        print(f"Error: {message}")
        Command_Engine.command_failed(viewer, f'Error: {message}')
        parent = getattr(viewer, 'main_window', None)
        if context is None:
            QMessageBox.critical(parent, translate("Viewer", "ESMFold Launch Error"), message.display())
        if hasattr(viewer, 'console_text'):
            failure = Message("Error: Could not launch the ESMFold terminal.")
            Command_Engine.show_status(viewer, failure)
        return
    except Exception as error:
        # No worker was started, so nothing else will delete the file.
        if tracker is not None:
            tracker.fail(str(error))
        _discard_worker_input(tmp_path)
        raise
    # A second esmfold is refused while this worker runs (step 4b).
    viewer.esmfold_worker = (process, tracker)

    # 10. Open Mol* web browser tab immediately
    page_opened = esmfold_backend.open_esmfold_ui(viewer, show_existing_dialog=False)
    mode_label = "Biohub ESM3" if is_large else "local ESM3"
    if is_large:
        spawning = Message("Spawning separate console to fold %n structure(s) with Biohub ESM3...", n=len(nodes_to_fold))
    else:
        spawning = Message("Spawning separate console to fold %n structure(s) with local ESM3...", n=len(nodes_to_fold))
    outcome = f"Started {mode_label} folding for {len(nodes_to_fold)} structure(s); waiting for the worker."
    if skipped_ids:
        skipped = Message("Skipped %n selected node(s) without a sequence.", n=len(skipped_ids))
        spawning = JoinedMessage([spawning, skipped])
        outcome += f" {skipped}"
        if len(skipped_ids) <= MAX_LISTED_SKIPPED:
            outcome += f" ({', '.join(skipped_ids)})"
    if clicked_note is not None:
        spawning = JoinedMessage([clicked_note, spawning])
        outcome += f" {clicked_note}"
    if not page_opened and not _structure_viewer_connected(viewer):
        outcome += " The structure viewer page was not opened; use the Fold View button if it does not appear."
    _set_console_text(viewer, spawning)
    Command_Engine.command_succeeded(viewer, outcome)
