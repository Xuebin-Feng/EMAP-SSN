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
import Command_Engine
from PySide6 import QtWidgets
from Viewer_Command_Portal import user_interaction, CURRENT

def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        msg = (
            "Usage: run\n"
            "Description: Opens a file explorer to select a command script file (.txt or .py) and executes the commands in sequence.\n"
            "  - For .txt files: Executes each line as a command.\n"
            "  - For .py files: Executes the Python script in a subprocess and runs the commands outputted to stdout.\n"
            "Example:\n"
            "  run"
        )
        Command_Engine.print_help(viewer, msg, report_message=False)
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    # Open the file explorer to select a file
    try:
        with user_interaction(viewer, "Select run file in the Viewer dialog"):
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                viewer.canvas.native,
                "Select Command File",
                "",
                "Command Scripts (*.txt *.py);;Text Files (*.txt);;Python Scripts (*.py);;All Files (*)"
            )
    except Exception as e:
        msg = f"Error opening file dialog: {e}"
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    if not file_path:
        msg = "File selection cancelled."
        Command_Engine.command_cancelled(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    if not os.path.exists(file_path):
        msg = f"Error: File '{file_path}' does not exist."
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    # Read/execute the file and extract commands
    try:
        commands_lines = []
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.py':
            import subprocess
            import sys
            from vispy import app as vispy_app
            
            print(f"[Run] Executing Python script: {file_path}")
            if hasattr(viewer, 'console_text'):
                viewer.console_text.text = "Executing Python script..."
                if hasattr(vispy_app, 'process_events'):
                    vispy_app.process_events()

            context = CURRENT.get()
            if context is not None:
                from Viewer_Worker_Tracking import ScriptTracker
                ScriptTracker(context, file_path)
                Command_Engine.command_succeeded(viewer, 'Python command script started; waiting for its output.')
                return

            # Execute python script in a subprocess using the current python executable
            result = subprocess.run(
                [sys.executable, file_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode != 0:
                stderr_output = result.stderr.strip()
                msg = f"Error: Python script failed (exit code {result.returncode}):\n{stderr_output}"
                Command_Engine.command_failed(viewer, msg)
                Command_Engine.print_help(viewer, msg)
                return
                
            commands_lines = result.stdout.splitlines()
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                commands_lines = f.readlines()

        context = CURRENT.get()
        if context is not None:
            commands = [line.split('//')[0].strip() for line in commands_lines]
            commands = [line for line in commands if line and line.split()[0].lower() != 'run']
            context.children(commands)
            Command_Engine.command_succeeded(viewer, f'Prepared {len(commands)} child commands.')
            return

        # Execute the commands in sequence
        executed_count = 0
        for line in commands_lines:
            # Split by // to remove any trailing comment
            cmd_line = line.split('//')[0].strip()
            if not cmd_line:
                continue
            
            parts = cmd_line.split()
            if not parts:
                continue
                
            command_name = parts[0].lower()
            if command_name == 'run':
                print("Warning: Recursive 'run' command in script ignored to prevent infinite loop.")
                continue
            
            print(f"[Run] Executing: {cmd_line}")
            Command_Engine._dispatch_user_command(viewer, cmd_line, record_history=False)
            executed_count += 1
            
        msg = f"Batch execution completed: {executed_count} commands run."
        Command_Engine.print_help(viewer, msg)
        
    except Exception as e:
        msg = f"Error reading/executing command file: {e}"
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return
    Command_Engine.command_succeeded(viewer, msg)
