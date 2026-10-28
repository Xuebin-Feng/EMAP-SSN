# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""console: open the Console window, which shows everything this Viewer's process has printed.

It runs at once, even while other commands wait (QUEUE below).
This module imports no Qt, so the VR viewer can reach it: the VR viewer has no
window of this kind, and the command says so there.
"""
import Command_Engine
from utilities.Localization import JoinedMessage, Message
from desktop.Command_Metadata import entry

# --- What this command declares about itself ---
# A window for the person at the Viewer: nothing in it an Agent can read.
AGENT_POLICY = 'refuse'

CATALOG_ENTRY = entry("Open the Console window, which shows everything this Viewer's process has printed since it started, with the time of each line and warnings and errors marked; the log file it names keeps every line. Needs a Viewer that shows windows: a headless or hidden one has none. Runs at once.")

# How the Viewer dispatches this command (Command_Declarations): it only opens a window and reads nothing
# a waiting command would change, so it runs at once.
QUEUE = 'instant'


def print_help():
    print("""
    Console: What This Viewer Has Printed
    =====================================
    Usage: console
           console help

    Description:
      Opens the Console window, or brings it forward if it is open. It shows
      everything this Viewer's process has printed since it started, as it
      arrives, with the time of each line: the commands' output and every
      warning and error. Errors are red and warnings amber. The window
      searches the output, filters it to warnings and errors, follows new
      lines while the last one is in view, copies and saves it, and opens the
      log file, which keeps every line of the session (the window shows the
      last 20,000). A second row under the search holds a button for each page
      you have opened from the Viewer, which opens that page again. The
      Console button at the bottom of the Viewer opens the same window.

    Notes:
      A Viewer that shows no window (headless, or hidden and not shown) has
      no Console window; read its log file, or use emapssn_viewer_data's log
      pages. A program started with --console prints to its terminal instead:
      its Console shows only the buttons, and says where the output is.

    Examples:
      console
    """)


def _refuse(viewer, message):
    Command_Engine.print_help(viewer, message, report_message=False)
    Command_Engine.command_failed(viewer, message)


def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the Console"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
        return
    if args:
        _refuse(viewer, JoinedMessage([
            Message("Error: console takes no arguments, not '{argument}'.", argument=args[0]),
            Message("Usage: {syntax}", syntax="console"),
        ], separator="\n"))
        return
    opener = getattr(viewer, 'open_console_window', None)
    # The desktop Viewer says "opened" (also when started with --console, which keeps no output: its Console
    # then holds the buttons and says where the output is) or "no_window" (headless, or hidden and not shown);
    # the VR viewer has no such window.
    result = opener() if opener is not None else "unavailable"
    if result == "opened":
        message = Message("Opened the Console.")
        Command_Engine.show_status(viewer, message)
        Command_Engine.command_succeeded(viewer, message)
    elif result == "no_window":
        _refuse(viewer, Message("The Console needs a window, and this Viewer shows none; its log file has the output."))
    else:
        _refuse(viewer, Message("The Console is not available: this Viewer has no window of that kind."))
