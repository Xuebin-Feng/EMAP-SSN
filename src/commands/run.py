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

import codecs
import io
import locale
import os
import Command_Engine
from PySide6 import QtWidgets
from desktop.Desktop_App import translate
from utilities.Localization import JoinedMessage, Message
from Viewer_Command_Portal import user_interaction, CURRENT

# A byte-order mark names a .txt file's encoding. UTF-32 LE's mark begins with
# UTF-16 LE's, so the UTF-32 marks are checked first.
BYTE_ORDER_MARKS = (
    (codecs.BOM_UTF32_LE, 'utf-32'), (codecs.BOM_UTF32_BE, 'utf-32'),
    (codecs.BOM_UTF8, 'utf-8-sig'),
    (codecs.BOM_UTF16_LE, 'utf-16'), (codecs.BOM_UTF16_BE, 'utf-16'),
)


def read_command_lines(file_path):
    """Lines of a .txt command file, decoded the way Notepad decodes it.

    A byte-order mark names the encoding. A file without one is UTF-8 if it
    decodes as UTF-8, and otherwise in the system's legacy encoding: the ANSI
    code page on Windows, older Notepad's and PowerShell 5.1 Set-Content's
    default. Bytes that still don't decode show as U+FFFD; dropping them would
    turn select "café" into select "caf", which also matches caffeine.
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    encoding = next((name for mark, name in BYTE_ORDER_MARKS if data.startswith(mark)), None)
    if encoding is None:
        try:
            data.decode('utf-8')
            encoding = 'utf-8'
        except UnicodeDecodeError:
            encoding = locale.getencoding()
            print(f"[Run] {file_path} is not UTF-8; reading it as {encoding}, "
                  "with U+FFFD for any byte that doesn't decode.")
    # Universal newlines, as readlines() on a text-mode file splits them.
    return io.StringIO(data.decode(encoding, errors='replace'), newline=None).readlines()


def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        # The console line shows the first line; the terminal shows it all, in English.
        msg = JoinedMessage([
            Message("Usage: {syntax}", syntax="run"),
            "Description: Opens a file explorer to select a command script file (.txt or .py) and executes the commands in sequence.\n"
            "  - For .txt files: Executes each line as a command.\n"
            "  - For .py files: Executes the Python script in a subprocess and runs the commands outputted to stdout.\n"
            "Example:\n"
            "  run",
        ], separator="\n")
        Command_Engine.print_help(viewer, msg, report_message=False)
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    # Open the file explorer to select a file
    try:
        with user_interaction(viewer, "Select run file in the Viewer dialog"):
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                viewer.canvas.native,
                translate("Viewer", "Select Command File"),
                "",
                ";;".join([
                    translate("Viewer", "Command Scripts (*.txt *.py)"),
                    translate("Viewer", "Text Files (*.txt)"),
                    translate("Viewer", "Python Scripts (*.py)"),
                    translate("Viewer", "All Files (*)"),
                ]),
            )
    except Exception as e:
        msg = Message("Error opening file dialog: {error}", error=e)
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    if not file_path:
        msg = Message("File selection cancelled.")
        Command_Engine.command_cancelled(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    if not os.path.exists(file_path):
        msg = Message("Error: File '{file}' does not exist.", file=file_path)
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
                Command_Engine.show_status(viewer, Message("Executing Python script..."))
                if hasattr(vispy_app, 'process_events'):
                    vispy_app.process_events()

            context = CURRENT.get()
            if context is not None:
                from Viewer_Worker_Tracking import ScriptTracker
                ScriptTracker(context, file_path)
                Command_Engine.command_succeeded(viewer, 'Python command script started; waiting for its output.')
                return

            # Execute python script in a subprocess using the current python executable.
            # Its output is decoded as UTF-8, so the child must print UTF-8; a piped
            # child otherwise uses the Windows ANSI code page. Bad bytes show as U+FFFD.
            # It has no stdin: a script calling input() would otherwise wait on the
            # Viewer's own stdin, and the Viewer with it.
            result = subprocess.run(
                [sys.executable, file_path],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            )
            
            if result.returncode != 0:
                stderr_output = result.stderr.strip()
                # The console line shows the first line; the script's errors are for the terminal.
                msg = JoinedMessage([
                    Message("Error: Python script failed (exit code {code}):", code=result.returncode),
                    stderr_output,
                ], separator="\n")
                Command_Engine.command_failed(viewer, msg)
                Command_Engine.print_help(viewer, msg)
                return
                
            commands_lines = result.stdout.splitlines()
        else:
            commands_lines = read_command_lines(file_path)

        context = CURRENT.get()
        if context is not None:
            commands = [Command_Engine.script_command(line) for line in commands_lines]
            commands = [line for line in commands if line and line.split()[0].lower() != 'run']
            context.children(commands)
            Command_Engine.command_succeeded(viewer, f'Prepared {len(commands)} child commands.')
            return

        # Execute the commands in sequence
        executed_count = 0
        failed_count = 0
        for line in commands_lines:
            # Drop blank lines, # lines and any trailing // comment
            cmd_line = Command_Engine.script_command(line)
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
            # Every line runs, as before; a line that reported a failure is counted.
            with Command_Engine.recorded_outcome() as outcome:
                Command_Engine._dispatch_user_command(viewer, cmd_line, record_history=False)
            executed_count += 1
            if outcome['status'] == 'failed':
                failed_count += 1

        if failed_count:
            msg = Message("Batch execution finished: {failed} of %n command(s) failed.",
                          n=executed_count, failed=failed_count)
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.print_help(viewer, msg, report_message=False)
            return

        msg = Message("Batch execution completed: %n command(s) run.", n=executed_count)
        Command_Engine.print_help(viewer, msg)
        
    except Exception as e:
        msg = Message("Error reading/executing command file: {error}", error=e)
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return
    Command_Engine.command_succeeded(viewer, msg)
