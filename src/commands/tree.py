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

"""
commands/tree.py — Thin CLI portal stub for Tree View, the guide tree of the loaded alignment.

Delegates what the page shows, and how it finds the tree, to web_ui/tree_backend.py.
"""

import Command_Engine
from Viewer_Command_Portal import CURRENT
from desktop.Command_Metadata import argument, choice, entry
from utilities.Localization import JoinedMessage, Message
from web_ui.Browser_Page import is_headless, page_address, page_is_open
from web_ui.tree_backend import locate_tree, register

# --- What this command declares about itself ---
# Opens a page in the user's browser, as `meta` and `esmfold` do, so an Agent reply that holds it waits for the
# user to press Run.
AGENT_POLICY = 'approve'

CATALOG_ENTRY = entry('Open Tree View in the browser: the guide tree the Embedding MSA tool saved for the alignment that is loaded, drawn beside the network. The page names no file: the tree is the one saved in the Guide_Trees folder of the alignment\'s folder, and an alignment that did not come from the tool has none. With no alignment, or no tree for it, the page still opens on a card that says why, and the result names the file it looked for. A page that is open already is not opened again, and the result says so. Needs a Viewer that shows windows.',
    argument('action', 'Omit to open the page. Registration only adds the Tree View button to the Console without opening it.', choice('--register-only')), opens_interface=True)

# The Agent's instructions for this command, numbered in the order given (desktop.Command_Metadata): after
# `esmfold` (240), before `jobs` (250).
AGENT_PROMPT_ORDER = 245
AGENT_PROMPT = """
`tree`
- Opens Tree View in the browser: a page that draws the guide tree the Embedding MSA tool saved for the loaded alignment, and adds its Console button. It takes no argument, and no file name: the page finds the tree itself, in the `Guide_Trees` folder beside the alignment, from the alignment's name.
- With no alignment loaded, an alignment the tool did not name, or no tree file, the page still opens on a card that says why, and the result names the file it looked for. The tree itself is not part of a snapshot, so what it shows is not something to describe to the user.
- A page that is open already is not opened again: the command succeeds and says so.
"""


# How the Viewer dispatches this command (Command_Declarations). Opening the page changes nothing a waiting
# command would see, so it runs at once. `tree --register-only` brings back the Tree View button of the Console
# when a session loads.
QUEUE = 'instant'
CONSOLE_BUTTON = True


def print_help():
    print("""
    Tree View: The Guide Tree of the Loaded Alignment
    =================================================
    Usage: tree
           tree help

    Description:
      Opens Tree View in your browser and adds its button to the Console. The page draws the guide
      tree that the Embedding MSA tool saved for the alignment loaded in this Viewer, and follows
      the alignment: load another one (the alignment command) and the page shows its tree.

    Notes:
      The tree is found from the alignment's name, never chosen: for Set_[model]_alignment.fasta it
      is Guide_Trees/Set_[model]_guide_tree.nex in the alignment's folder. An alignment that did not
      come from the tool, a renamed one, or one made before trees were saved has none, and the page
      says which file it looked for.

    Examples:
      tree
    """)


def _refuse(viewer, message):
    Command_Engine.print_help(viewer, message, report_message=False)
    Command_Engine.command_failed(viewer, message)


def _no_tree_note(viewer):
    """The sentence that says why Tree View has no tree to show, or None when it has one (or something that may be one).

    The page opens all the same: a request that is valid succeeds, with this to tell the person what is missing.
    """
    found = locate_tree(viewer)
    if found["status"] == "no_alignment":
        return Message("No alignment is loaded, so there is no guide tree to show.")
    if found["status"] == "unnamed_alignment":
        return Message("The loaded alignment, {alignment}, was not named by the Embedding MSA tool, so it has no guide tree.",
                       alignment=found["alignment"])
    if found["status"] == "no_tree":
        return Message("No guide tree was found for the loaded alignment; Tree View looks for {path}.", path=found["tree_path"])
    return None


def _succeeded(viewer, message):
    """Report success with ``message``, and after it the note on a missing tree when there is one."""
    note = _no_tree_note(viewer)
    Command_Engine.command_succeeded(viewer, message if note is None else JoinedMessage([message, note], separator=" "))


def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the Console"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
        return
    if args and args != ['--register-only']:
        _refuse(viewer, JoinedMessage([
            Message("Error: tree takes no arguments, not '{argument}'.", argument=args[0]),
            Message("Usage: {syntax}", syntax="tree"),
        ], separator="\n"))
        return

    register(viewer)
    if args:
        Command_Engine.command_succeeded(viewer, Message("Registered Tree View."))
        return

    # A command from the MCP or the agent has nobody at the Viewer to dismiss a dialog, so an already-open page
    # is reported on the status line only.
    options = {"show_existing_dialog": False} if CURRENT.get() is not None else {}
    if viewer.open_page("tree", **options):
        _succeeded(viewer, Message("Opened Tree View."))
    elif page_is_open(viewer, "tree"):
        # Nothing to open: it is open. A valid request that changes nothing succeeds, with a note.
        _succeeded(viewer, Message("Tree View is already open in your browser."))
    elif is_headless(viewer) and (address := page_address(viewer, "/tree.html")) is not None:
        _succeeded(
            viewer,
            Message("A Viewer with no window opens no browser page; Tree View is served at {address}.", address=address),
        )
    else:
        Command_Engine.command_failed(viewer, Message("Could not open Tree View."))
