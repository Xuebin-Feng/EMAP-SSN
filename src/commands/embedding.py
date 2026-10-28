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
import EMAPSSN_Config as cfg
import Command_Engine
from PySide6 import QtWidgets
from desktop.Desktop_App import translate
from utilities.Localization import JoinedMessage, Message
from Viewer_Command_Portal import user_interaction

def print_help():
    print("""
    Embedding Switcher Tool
    =======================
    Usage:
      embedding
          Opens a file explorer to select an embedding file (.h5) manually.
      embedding [filename.h5]
          Loads the specified embedding file directly from disk.
      embedding help
          Displays this help message.

    Loading Rules:
      The embedding may contain all, some, or none of the nodes currently plotted in the viewer.
      Missing nodes remain visible but are excluded from embedding-dependent analyses.
      Coverage is matched by exact full headers and identical sequences, and is reported when the embedding loads.
      Unreadable, malformed or incomplete files still fail and restore the previous embedding.
      The switch lasts for this session and is not an undo step; the saved settings are unchanged.
    """)


def refresh_session_card(viewer):
    """Show the active embedding on the Viewer's info card, if it is open."""
    refresh = getattr(viewer, '_refresh_session_card', None)
    if callable(refresh):
        refresh()


# The Session card's file this command picks: the button beside it types this command
# (Command_Declarations.session_file_pickers).
PICKS_SESSION_FILE = 'embedding'


def run(viewer, args):
    # Check for help argument
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the Console"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
        return

    embed_dir = getattr(cfg, 'EMBED_DIR', "Embeddings")

    # 1. No arguments provided -> Open File Explorer for manual selection
    if not args:
        try:
            with user_interaction(viewer, "Select embedding file in the Viewer dialog"):
                file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    viewer.canvas.native,
                    translate("Viewer", "Select Embedding File"),
                    embed_dir if os.path.exists(embed_dir) else "",
                    ";;".join([
                        translate("Viewer", "Embedding Files (*.h5)"),
                        translate("Viewer", "All Files (*)"),
                    ]),
                )
        except Exception as e:
            msg = Message("Error opening file dialog: {error}", error=e)
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.print_help(viewer, msg)
            return

        if not file_path:
            msg = Message("Embedding selection cancelled.")
            Command_Engine.command_cancelled(viewer, msg)
            Command_Engine.print_help(viewer, msg)
            return

        if not os.path.exists(file_path):
            msg = Message("Error: File '{file}' does not exist.", file=file_path)
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.print_help(viewer, msg)
            return

        selected_file = os.path.basename(file_path)
        new_path = file_path.replace("\\", "/")
    else:
        # Treat args as target direct file path
        identifier = " ".join(args).strip()
        # A quoted path, such as one with spaces, keeps its quotes in args.
        if len(identifier) >= 2 and identifier[0] == identifier[-1] and identifier[0] in "\"'":
            identifier = identifier[1:-1]
        selected_file = None
        new_path = None

        # 1. Check if the identifier is a direct file path
        if os.path.exists(identifier) and os.path.isfile(identifier):
            new_path = os.path.abspath(identifier).replace("\\", "/")
            selected_file = os.path.basename(identifier)
        else:
            # 2. Check if it exists in the embedding directory directly
            path_in_dir = os.path.join(embed_dir, identifier)
            if os.path.exists(path_in_dir) and os.path.isfile(path_in_dir):
                new_path = os.path.abspath(path_in_dir).replace("\\", "/")
                selected_file = os.path.basename(path_in_dir)
            else:
                # 3. Try the .h5 extension if missing
                for ext in ['.h5']:
                    if os.path.exists(identifier + ext) and os.path.isfile(identifier + ext):
                        new_path = os.path.abspath(identifier + ext).replace("\\", "/")
                        selected_file = os.path.basename(identifier + ext)
                        break
                    elif os.path.exists(os.path.join(embed_dir, identifier + ext)) and os.path.isfile(os.path.join(embed_dir, identifier + ext)):
                        new_path = os.path.abspath(os.path.join(embed_dir, identifier + ext)).replace("\\", "/")
                        selected_file = os.path.basename(os.path.join(embed_dir, identifier + ext))
                        break

        if not new_path:
            msg = Message(
                "Error: Embedding file '{file}' not found (checked absolute, relative, and {folder}).",
                file=identifier,
                folder=embed_dir,
            )
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.print_help(viewer, msg)
            return

    # Load the selected embedding file
    print(f"\nAttempting to load embedding: {selected_file}...")
    if hasattr(viewer, 'console_text'):
        Command_Engine.show_status(viewer, Message("Loading {file}...", file=selected_file))

    # Backup current state for safety rollback
    backup_embed_file = cfg.EMBED_FILE
    backup_embedding = getattr(viewer, 'embedding', None)

    cfg.EMBED_FILE = new_path
    reason = None

    try:
        viewer.load_embedding()

        # A parseable partial or zero-overlap embedding is valid. Only a file the
        # manager refuses leaves it without a source and triggers rollback. The
        # manager says why on the object it leaves behind.
        embedding = viewer.embedding
        if embedding is None or getattr(embedding, 'source', None) is None:
            reason = getattr(embedding, 'load_failure', None)
            raise ValueError(
                "Embedding loader failed to return an embedding." if reason is None else reason
            )

        embedded_count = len(getattr(embedding, 'matched_headers', []))
        total_count = len(getattr(viewer, 'full_headers', []))
        # What follows the coverage, on the console line translated and in the
        # terminal and the command's record in English: the nodes whose
        # sequence differs, the embedding's model, and a network built with
        # another model.
        details = embedding.describe(getattr(cfg, 'INPUT_HDF5', ''))

        success_msg = (
            f"Success: Loaded embedding '{selected_file}' "
            f"({embedded_count}/{total_count} network nodes embedded)."
        )
        print(f"\n{success_msg}")
        for detail in details:
            print(detail)
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, JoinedMessage([
                Message(
                    "Loaded {file}: {embedded}/{total} embedded.",
                    file=selected_file, embedded=embedded_count, total=total_count,
                ),
                *details,
            ]))

        refresh_session_card(viewer)
        Command_Engine.command_succeeded(viewer, " ".join([success_msg, *map(str, details)]))

    except Exception as e:
        print(f"\nFailed to load embedding '{selected_file}': {e}")
        print("Reverting to previous embedding state...")

        # Rollback
        cfg.EMBED_FILE = backup_embed_file
        viewer.embedding = backup_embedding
        refresh_session_card(viewer)

        # The console line shows it translated; the command's record keeps the English.
        failure = Message("Load failed. Reverted to previous embedding.")
        if hasattr(viewer, 'console_text'):
            # The loader's reason, when it gave one, follows on the console line;
            # the record of the failure holds it too.
            Command_Engine.show_status(viewer, failure if reason is None else JoinedMessage([failure, reason]))
        Command_Engine.command_failed(
            viewer, f"Failed to load embedding '{selected_file}': {e}\nReverted to previous embedding."
        )
