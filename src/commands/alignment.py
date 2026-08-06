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
import fnmatch
import SSN_Config as cfg
import Command_Engine
from PySide6 import QtWidgets

def print_help():
    print("""
    Alignment Switcher Tool
    =======================
    Usage:
      alignment
          Opens a file explorer to select an MSA file (.fasta or .h5) manually.
      alignment [filename.fasta / filename.h5]
          Loads the specified MSA file directly from disk.
      alignment help
          Displays this help message.

    Loading Rules:
      To load successfully, the FASTA subset currently representing the nodes in the viewer 
      must be a strict subset of the sequence headers present in the new MSA file. 
      If any node is missing, the load fails and the system automatically rolls back 
      to the previously active alignment to ensure session stability.
    """)

def run(viewer, args):
    # Check for help argument
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    msa_dir = getattr(cfg, 'MSA_DIR', os.path.join("Input_Files", "Multiple_Alignments"))
    
    # 1. No arguments provided -> Open File Explorer for manual selection
    if not args:
        try:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                viewer.canvas.native,
                "Select Alignment File",
                msa_dir if os.path.exists(msa_dir) else "",
                "Alignment Files (*.fasta *.h5);;FASTA Files (*.fasta);;HDF5 Files (*.h5);;All Files (*)"
            )
        except Exception as e:
            msg = f"Error opening file dialog: {e}"
            Command_Engine.print_help(viewer, msg)
            return

        if not file_path:
            msg = "Alignment selection cancelled."
            Command_Engine.print_help(viewer, msg)
            return

        if not os.path.exists(file_path):
            msg = f"Error: File '{file_path}' does not exist."
            Command_Engine.print_help(viewer, msg)
            return

        selected_file = os.path.basename(file_path)
        new_path = file_path.replace("\\", "/")
    else:
        # Treat args as target direct file path
        identifier = " ".join(args).strip()
        selected_file = None
        new_path = None
        
        # 1. Check if the identifier is a direct file path
        if os.path.exists(identifier) and os.path.isfile(identifier):
            new_path = os.path.abspath(identifier).replace("\\", "/")
            selected_file = os.path.basename(identifier)
        else:
            # 2. Check if it exists in the msa directory directly
            path_in_dir = os.path.join(msa_dir, identifier)
            if os.path.exists(path_in_dir) and os.path.isfile(path_in_dir):
                new_path = os.path.abspath(path_in_dir).replace("\\", "/")
                selected_file = os.path.basename(path_in_dir)
            else:
                # 3. Try common extensions if missing (.fasta, .h5)
                for ext in ['.fasta', '.h5']:
                    if os.path.exists(identifier + ext) and os.path.isfile(identifier + ext):
                        new_path = os.path.abspath(identifier + ext).replace("\\", "/")
                        selected_file = os.path.basename(identifier + ext)
                        break
                    elif os.path.exists(os.path.join(msa_dir, identifier + ext)) and os.path.isfile(os.path.join(msa_dir, identifier + ext)):
                        new_path = os.path.abspath(os.path.join(msa_dir, identifier + ext)).replace("\\", "/")
                        selected_file = os.path.basename(os.path.join(msa_dir, identifier + ext))
                        break
                        
        if not new_path:
            msg = f"Error: Alignment file '{identifier}' not found (checked absolute, relative, and {msa_dir})."
            Command_Engine.print_help(viewer, msg)
            return

    # Load the selected alignment file
    print(f"\nAttempting to load alignment: {selected_file}...")
    if hasattr(viewer, 'console_text'):
        viewer.console_text.text = f"Loading {selected_file}..."

    # Backup current state for safety rollback
    backup_msa_file = cfg.MSA_FILE
    backup_alignment = viewer.alignment
    backup_active_ref = viewer.active_reference

    cfg.MSA_FILE = new_path

    try:
        viewer.load_global_alignment()

        # Check for missing reference sequence fallback
        if viewer.alignment is None or viewer.alignment.aln is None:
            print(f"Warning: Active reference '{viewer.active_reference}' not found in '{selected_file}'.")
            print("Re-attempting load in Pure Occupancy Mode (no reference sequence)...")
            viewer.active_reference = None
            viewer.load_global_alignment()

            if viewer.alignment is None or viewer.alignment.aln is None:
                raise ValueError("Alignment loader failed to return an alignment.")
            else:
                success_msg = f"Success: Loaded '{selected_file}' in Pure Occupancy Mode."
                print(f"\n{success_msg}")
                if hasattr(viewer, 'console_text'):
                    viewer.console_text.text = f"Loaded {selected_file} (No Ref)"
        else:
            success_msg = f"Success: Loaded alignment '{selected_file}'."
            print(f"\n{success_msg}")
            if hasattr(viewer, 'console_text'):
                viewer.console_text.text = f"Loaded {selected_file}"

    except Exception as e:
        # Check if it was a subset violation error
        err_msg = str(e)
        if "subset" in err_msg.lower() or "missing" in err_msg.lower():
            explanation = (
                "\nCRITICAL ERROR: Sequence subset violation!\n"
                "The active network sequence set (FASTA subset) must be a strict subset of the sequences in the MSA.\n"
                "One or more nodes in the current view do not exist in the new alignment."
            )
            print(explanation)
        
        print(f"\nFailed to load alignment '{selected_file}': {e}")
        print("Reverting to previous alignment state...")
        
        # Rollback
        cfg.MSA_FILE = backup_msa_file
        viewer.alignment = backup_alignment
        viewer.active_reference = backup_active_ref
        
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Load failed. Reverted to previous alignment."
