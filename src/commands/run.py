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
from Viewer_Command_Portal import user_interaction, CURRENT, MAX_SCRIPT_COMMANDS
from desktop.Command_Metadata import argument, choice, entry

# --- What this command declares about itself ---
# Opens a dialog and runs a script.
AGENT_POLICY = 'approve'

CATALOG_ENTRY = entry('Open a script-selection dialog and execute a text command batch or Python-generated commands, stopping at the first command that fails.',
    argument('script', 'Chosen through a visible Viewer dialog, not a path argument. TXT lines become commands; Python stdout supplies commands. Nested run lines are ignored. A Python script runs in its own folder. The first command that fails or is cancelled ends the run and the later commands are not run; a script of more than 1000 commands is refused before any runs. MCP tracks child commands and Python workers; typed in the Viewer, a Python script runs in the background, one at a time.'),
    argument('stop', 'Optional word stop: ends the Python scripts the Viewer is running, with every process they started, and runs none of their commands; with none running it succeeds and says so. No dialog opens. A request holding only stop commands runs at once, even while the script request it ends still waits. Closing the Viewer stops them too.', choice('stop')), writes_files=True, opens_interface=True)

# The Agent's instructions for this command, numbered in the order given (desktop.Command_Metadata).
AGENT_PROMPT_ORDER = 90
AGENT_PROMPT = r"""
`run`
- Opens a file chooser for a `.txt` command script or a `.py` command-generating script. It does not accept a filepath argument.
- Text scripts execute each nonblank line after removing a trailing `//` comment. Python scripts run in a subprocess and each nonblank stdout line is treated as a viewer command. A script line ending in ` \` continues on the next; a reply line never does.
- Recursive `run` lines are ignored to prevent loops.
- A Python script typed into the Viewer runs in the background, one at a time, and its commands run when it ends.
- `run stop` stops the Python scripts the Viewer is running, with every process they started, and none of their commands run; with none running it reports that and succeeds. Closing the Viewer stops them too.
- The run stops at the first command that fails or is cancelled: the commands after it are not run.
"""

# How the Viewer dispatches this command (Command_Declarations). run opens its dialog at once, and the
# commands of the script then wait their turn like any other.
QUEUE = 'instant'


def acts_on_jobs(args):
    """`run stop` ends the Python scripts that run, which the request at the head of the portal's line may wait for."""
    return [word.lower() for word in args] == ['stop']


def is_run_command(command_line):
    """Whether a script line is itself a run command, which a script's commands may not hold."""
    return command_line.split()[0].lower() == 'run'

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


def script_commands(command_lines):
    """The commands a script's lines hold: continued lines are joined, and blank and comment lines, and nested run lines, are left out.

    Raises Command_Engine.ScriptContinuationError, before any command is
    returned, for a script whose last line continues.
    """
    commands = []
    # Joins the lines a backslash continues, and drops blank lines, # lines and any trailing // comment.
    for cmd_line in Command_Engine.script_command_lines(command_lines):
        if is_run_command(cmd_line):
            print("Warning: Recursive 'run' command in script ignored to prevent infinite loop.")
            continue
        commands.append(cmd_line)
    return commands


def refuse_script(viewer, error):
    """Report a script whose last line continues past its end: it is refused as a whole, and no command runs."""
    msg = error.args[0]
    Command_Engine.command_failed(viewer, msg)
    Command_Engine.print_help(viewer, msg, report_message=False)


def run_script_lines(viewer, command_lines):
    """Run a typed script's command lines in order, and report how it went.

    The rules are those of a script sent through the command portal: a script
    holds at most MAX_SCRIPT_COMMANDS commands, and the first command that
    fails, or is cancelled, ends the run. The later commands often depend on
    it, so none of them runs.

    In a Viewer with a command queue, a command that has to wait its turn (one
    given while another computes, and every one after it) is queued; the queue
    then cancels the script's later commands if it fails (Command_Queue).
    """
    try:
        commands = script_commands(command_lines)
    except Command_Engine.ScriptContinuationError as error:
        refuse_script(viewer, error)
        return
    total = len(commands)
    if total > MAX_SCRIPT_COMMANDS:
        msg = Message("Error: The script holds {count} commands, more than the {limit} a script may run. None were run.",
                      count=total, limit=MAX_SCRIPT_COMMANDS)
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg, report_message=False)
        return

    queue = getattr(viewer, 'command_queue', None)
    waiting = 0
    for index, cmd_line in enumerate(commands, 1):
        print(f"[Run] Executing: {cmd_line}")
        with Command_Engine.recorded_outcome() as outcome, Command_Engine.typed_script():
            Command_Engine._dispatch_user_command(viewer, cmd_line, record_history=False)
        if outcome['status'] is None and queue is not None and queue.pending():
            waiting += 1
            continue
        if outcome['status'] not in ('failed', 'cancelled'):
            continue
        if outcome['status'] == 'failed':
            stopped = Message("Batch stopped at command {index} of {total}: '{command}' failed.",
                              index=index, total=total, command=cmd_line)
        else:
            stopped = Message("Batch stopped at command {index} of {total}: '{command}' was cancelled.",
                              index=index, total=total, command=cmd_line)
        remaining = total - index
        msg = stopped if not remaining else JoinedMessage([
            stopped, Message("Not run: the remaining %n command(s).", n=remaining)])
        if outcome['status'] == 'failed':
            Command_Engine.command_failed(viewer, msg)
        else:
            Command_Engine.command_cancelled(viewer, msg)
        Command_Engine.print_help(viewer, msg, report_message=False)
        return

    if waiting:
        msg = Message("Batch given: %n command(s). Waiting their turn: {waiting}; the first of them that fails "
                      "cancels the ones after it.", n=total, waiting=waiting)
    else:
        msg = Message("Batch execution completed: %n command(s) run.", n=total)
    Command_Engine.print_help(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)


def finish_script(viewer, result):
    """What the end of a typed Python script brings, on the GUI thread: its failure, or its commands run.

    result is the script's CompletedProcess, or the exception that kept it from running.
    """
    try:
        if isinstance(result, Exception):
            raise result
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
        run_script_lines(viewer, result.stdout.splitlines())
    except Exception as e:
        msg = Message("Error reading/executing command file: {error}", error=e)
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)


def stop_scripts(viewer):
    """`run stop`: end the Python command scripts the Viewer is running, and every process they started.

    Their commands are not run, nor are the commands of typed scripts that
    still wait their turn in the command queue. With none of either, it is a
    valid no-op that says so.
    """
    from Viewer_Worker_Tracking import stop_command_scripts
    count = stop_command_scripts()
    queue = getattr(viewer, 'command_queue', None)
    lines = [job for job in queue.pending() if job.source == 'script'] if queue is not None else []
    for job in lines:
        queue.cancel(job, Message("Cancelled by run stop."))
    if count:
        msg = Message("Stopped %n Python script(s).", n=count)
    else:
        msg = Message("No Python script is running.")
    if lines:
        msg = JoinedMessage([msg, Message("Cancelled %n waiting script command(s).", n=len(lines))])
    if hasattr(viewer, 'console_text'):
        Command_Engine.show_status(viewer, msg)
    print(msg)
    Command_Engine.command_succeeded(viewer, msg)


def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        # The console line shows the first line; the terminal shows it all, in English.
        msg = JoinedMessage([
            Message("Usage: {syntax}", syntax="run [stop]"),
            "Description: Opens a file explorer to select a command script file (.txt or .py) and executes the commands in sequence.\n"
            "  - For .txt files: Executes each line as a command; text after // is a comment.\n"
            "  - For .py files: Executes the Python script in a subprocess, in the script's own folder and in the background, "
            "and then runs the commands outputted to stdout. One script runs at a time.\n"
            "  - A command may go on over several lines: a line that ends in a space and a backslash continues on the next "
            "line, which may start with a #label#. The same goes for the lines a Python script prints.\n"
            "  - run stop: Stops the Python scripts the Viewer is running, and every process they started; "
            "their commands are not run. Closing the Viewer stops them too.\n"
            "  - The first command that fails stops the run; the commands after it are not run.\n"
            f"  - A script may hold at most {MAX_SCRIPT_COMMANDS} commands; a longer one is refused before any command runs. "
            "So is one whose last line ends in a continuing backslash.\n"
            "Examples:\n"
            "  run\n"
            "  run stop",
        ], separator="\n")
        Command_Engine.print_help(viewer, msg, report_message=False)
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
        return
    if args and args[0].lower() == 'stop':
        stop_scripts(viewer)
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
        _, ext = os.path.splitext(file_path)
        context = CURRENT.get()

        if ext.lower() == '.py':
            print(f"[Run] Executing Python script: {file_path}")
            if context is None and getattr(viewer, 'command_script_run', None) is not None:
                # One typed script at a time: the Viewer would otherwise run the commands of two
                # scripts in the order they happen to finish.
                msg = Message("A Python script is already running; wait for it to finish before running another.")
                Command_Engine.command_failed(viewer, msg)
                Command_Engine.print_help(viewer, msg, report_message=False)
                return

            msg = Message("Executing Python script...")
            if hasattr(viewer, 'console_text'):
                Command_Engine.show_status(viewer, msg)

            if context is not None:
                from Viewer_Worker_Tracking import ScriptTracker
                ScriptTracker(context, file_path, keep_command=lambda line: not is_run_command(line))
                Command_Engine.command_succeeded(viewer, 'Python command script started; waiting for its output.')
                return

            # A typed script runs in the background, so the Viewer stays usable, and its commands run
            # on the GUI thread when it ends (finish_script). It runs as ScriptTracker runs one for the
            # command portal: with no stdin, printing UTF-8, and in the script's own folder.
            from Viewer_Worker_Tracking import TypedScriptRun
            TypedScriptRun(viewer, file_path, finish_script)
            Command_Engine.command_succeeded(viewer, msg)
            return

        commands_lines = read_command_lines(file_path)

        if context is not None:
            try:
                commands = [line for line in Command_Engine.script_command_lines(commands_lines)
                            if not is_run_command(line)]
            except Command_Engine.ScriptContinuationError as error:
                refuse_script(viewer, error)
                return
            context.children(commands)
            Command_Engine.command_succeeded(viewer, f'Prepared {len(commands)} child commands.')
            return

        run_script_lines(viewer, commands_lines)

    except Exception as e:
        msg = Message("Error reading/executing command file: {error}", error=e)
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return
