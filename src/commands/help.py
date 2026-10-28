# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""help: list the commands, or show the help of one, as ``<command> help`` does.

``help`` alone lists every command with a one-line summary. ``help <command>``
prints that command's help exactly as ``<command> help`` does, and does not run
the command: it calls the module's run(viewer, ['help']), the form every
command answers with its help (Command_Declarations; the dispatch runs it at
once for every command). ``help help`` prints this command's own help.

The listing is read from the catalog the MCP server's get_command_catalog
serves (desktop.Viewer_Inspection.command_catalog), so the two cannot drift,
and a plugin dropped into src/commands is listed with no change here. The VR
viewer has a help command of its own (opt_vr), which describes its commands.

It changes nothing, so it runs at once, even while other commands wait (QUEUE
below), and it imports no Qt.
"""
import re
import textwrap

import Command_Engine
from utilities.Localization import JoinedMessage, Message
from desktop.Command_Metadata import argument, entry

# --- What this command declares about itself ---
# Harmless, but the Agent's prompt already holds every command's usage, so the output would only fill its
# context (a small model's is about 4,000 tokens): the prompt has no item for it and names it as refused.
AGENT_POLICY = 'refuse'

CATALOG_ENTRY = entry("List the commands, or print one command's help as <command> help does, without running it. Run help help for this command's own help. It runs at once and changes nothing.",
    argument('command', 'Optional command name, in any letter case: omit it to list the commands. A name that is no command fails and nothing else happens; a second word is refused.'))

# How the Viewer dispatches this command (Command_Declarations): it reads no state a waiting command would
# change, so it runs at once.
QUEUE = 'instant'

# The words that ask a command for its help (the engine's own list); each asks for this command's own help too.
HELP_WORDS = ('help', '-h', '--help', '-?')

# The listing's lines are wrapped at this width, the summaries hanging under their column.
LISTING_WIDTH = 110
# The least a clause before a semicolon needs to say what the command does, in characters.
MIN_CLAUSE = 20


def print_help():
    print("""
    Help: List the Commands, or Show the Help of One
    ================================================
    Usage: help
           help <command>
           help help

    Description:
      help alone lists every command with a one-line summary. help <command>
      prints that command's help, exactly as <command> help does, without
      running the command. The summaries are the ones the MCP server's command
      catalog gives. A command name is not case-sensitive, and a word that is no
      command fails.

    Examples:
      help
      help zoom
      help help
    """)


def _refuse(viewer, message):
    Command_Engine.print_help(viewer, message, report_message=False)
    Command_Engine.command_failed(viewer, message)


def _printed(viewer):
    if hasattr(viewer, 'console_text'):
        Command_Engine.show_status(viewer, Message("Help information printed to the Console"))
    Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')


def one_line(summary):
    """What a catalog summary says first: its first sentence, up to a semicolon that follows a whole clause.

    The summaries are written for the MCP catalog, where a clause after the semicolon names what the command
    needs or how it ends; a listing wants only what it does.
    """
    text = ' '.join(summary.split())
    first = re.split(r'(?<=\.)\s+(?=[A-Z])', text, maxsplit=1)[0]
    clause, semicolon, _rest = first.partition(';')
    return clause if semicolon and len(clause) >= MIN_CLAUSE else first.rstrip('.')


def listing(commands):
    """The text that lists ``commands``, the entries of the command catalog, one name and summary each."""
    width = max((len(entry['command']) for entry in commands), default=0)
    indent = ' ' * (2 + width + 2)
    lines = ['', 'EMAP-SSN Viewer commands', '', "  Type help <command>, or <command> help, for a command's usage and examples.", '']
    for entry in commands:
        wrapped = textwrap.wrap(one_line(entry['summary']), width=max(LISTING_WIDTH - len(indent), 30)) or ['']
        lines.append(f"  {entry['command'].ljust(width)}  {wrapped[0]}")
        lines.extend(indent + more for more in wrapped[1:])
    return '\n'.join(lines)


def _list_commands(viewer):
    from desktop.Viewer_Inspection import command_catalog
    print(listing(command_catalog()['commands']))
    _printed(viewer)


def _command_module(viewer, name):
    """The module of command ``name``, found as the dispatch finds it, or None when there is no such command.

    A private module (a name that starts with an underscore) is no command, as the catalog says, and neither
    is a module without run(viewer, args). A command whose module cannot be imported raises, as typing the
    command would.
    """
    if not name.isidentifier() or name.startswith('_'):
        return None
    try:
        module = Command_Engine.load_command_module(viewer, name)
    except ModuleNotFoundError as error:
        if error.name == f'commands.{name}':
            return None
        raise
    return module if hasattr(module, 'run') else None


def _unknown(viewer, name):
    failure = f"Unknown command: {name}"
    # The status line has the translated short form; the terminal, the Console and the log have the red line.
    Command_Engine.print_help(viewer, Message("Unknown command: {command}", command=name),
                              terminal_msg=f"Error: {failure}. Type help for the list of commands.",
                              report_message=False)
    Command_Engine.command_failed(viewer, failure)


def run(viewer, args):
    if not args:
        _list_commands(viewer)
        return
    if len(args) > 1:
        _refuse(viewer, JoinedMessage([
            Message("Error: help takes at most one command name, not '{argument}'.", argument=args[1]),
            Message("Usage: {syntax}", syntax="help [command]"),
        ], separator="\n"))
        return
    name = args[0].lower()
    if name in HELP_WORDS:
        print_help()
        _printed(viewer)
        return
    module = _command_module(viewer, name)
    if module is None:
        _unknown(viewer, name)
        return
    # The command is not run: only its help, the form every command answers (see the module's docstring).
    module.run(viewer, ['help'])
