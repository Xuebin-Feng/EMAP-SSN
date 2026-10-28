# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""What a command module declares about how the Viewer dispatches it.

A command is a module of src/commands, with run(viewer, args). The Viewer's
dispatch (Command_Engine), its command queue (Command_Queue) and the command
portal that serves MCP and the Agent page (Viewer_Command_Portal) never name a
command. Where one has to be treated differently from the rest, they ask its
module, and a module that says nothing is treated cautiously. A new command, or
a plugin dropped into src/commands, is therefore dispatched correctly without
any change to the program. The few places where central code still holds a
command's name are listed, each with its reason, in ALLOWED of
tests/test_command_declarations.py (CentralCodeNamesNoCommandTests), which fails
when any other file of the program names a command, imports one or tabulates them.
The docstring of the commands package (src/commands/__init__.py) lists every
declaration a module can make and where each is read.

A module may declare any of the following at its top level. ``args`` is a list
of the words after the command's name, as they were given (a function compares
``word.lower()`` where case does not matter); ``source`` is where the command
comes from: 'typed' at the Viewer's command line, 'script' for a line of a
script typed there (run), 'portal' for MCP and the Agent page.

* Whether the command waits its turn in the command queue, or runs at once
  even while other commands wait. Default: it waits ('ordered').

      QUEUE = 'instant'                     # a constant: 'instant' or 'ordered'

      def queue_kind(args, source):         # or a function of the arguments
          return 'instant' if ... else 'ordered'

  A command is 'instant' when it reads nothing a waiting command would change
  or acts on the commands that wait (the camera, the job commands, a stop).
  Help (help, -h, --help, -?) and --register-only run at once for every command;
  the engine decides that, and a module need not.

* Whether the command starts a request to a model. A command that a model
  generated (from the portal) is refused when this is true, so that a model
  cannot start another model request. Default: it does not.

      STARTS_MODEL_REQUEST = True           # a constant

      def starts_model_request(args):       # or a function of the arguments
          return ...

* Whether the command acts on the jobs that run, the commands that wait or the
  work they started: a stop command or a job list. The command portal runs a
  request that holds only such commands at once, ahead of the request at the
  head of its line, which may be waiting for those very jobs. Default: it does
  not.

      ACTS_ON_JOBS = True                   # a constant

      def acts_on_jobs(args):               # or a function of the arguments
          return ...

* Whether the command puts a button in the Console window's second row, which comes
  back when a session that had it is loaded. The Viewer saves the commands that
  registered one with the session and, when it opens a layout cache, runs
  ``<command> --register-only`` for each of them. A cache is a file people share,
  so what it lists is run only for a command whose module declares this. Default:
  it does not.

      CONSOLE_BUTTON = True                 # a constant, and only a constant

* Which of the Session card's files the command picks. The card, which the Viewer's
  info button opens, lists the files the session was built from and puts a button
  beside each that the Viewer lets it change; the button types the command that picks
  another file, with no argument, as if it were typed. The Viewer keeps the file keys
  that can be changed ('msa', 'embedding') and the buttons' tooltips; the command says
  which of them it picks, and the card never names the command. Default: it picks
  none, and a file no command picks has no button.

      PICKS_SESSION_FILE = 'msa'            # a constant, and only a constant: the card's key of the file

  session_file_pickers() reads it over the commands folder: {file key: command}. Two
  commands that declare the same file make a button that could type either, so that
  file has none, and a note on stderr names the commands.

* What the command resets, and what `reset help` says of it. ``RESET_TARGETS = ('cluster',)``
  with ``reset(viewer, target)``, and optionally ``reset_changes_state(viewer, target)`` and
  ``RESET_HELP = {'clusters': (40, 'what it restores')}``, are not read here: commands/reset.py
  imports the modules that mention RESET_TARGETS, and its docstring says what each means.

A function wins over the constant of the same declaration. A value that is not
what the declaration asks for, or a function that raises, counts as undeclared
(ordered; not acting on jobs; no Console button; picking no file), except for STARTS_MODEL_REQUEST:
a declaration that cannot be read counts as true there, so that a mistake refuses
the command rather than lets it start a request. A module that cannot be imported
has declared nothing, and its command fails when it is run, as it did before.

A module also declares what the command catalog and the Agent say about it: CATALOG_ENTRY
(its summary and arguments), AGENT_POLICY (whether an Agent reply may run it at once, needs the
user's approval or is refused), AGENT_PROMPT and AGENT_PROMPT_ORDER (its item in the Agent's
instructions). Those are not read by importing the module, as the ones above are, but from its
source, as literals that nothing in the module can change when it runs: desktop.Command_Metadata
says how, and what stands in for one that is missing or cannot be read ('approve' for the policy,
never 'run').
"""
import importlib
import sys
from pathlib import Path

INSTANT = 'instant'
ORDERED = 'ordered'

# What _declared gives for a module that makes no such declaration, and for one that cannot be used (a
# function that raised, or a name that is not a function).
_UNDECLARED = object()
_UNUSABLE = object()


def command_module(name):
    """The module of command ``name`` (commands.<name>) as it is loaded now, or None.

    None when there is no such command or its module cannot be imported. The
    module is imported if need be and never reloaded: the dispatch reloads the
    module of the command it runs, and this only reads declarations.
    """
    if not isinstance(name, str) or not name.isidentifier():
        return None
    try:
        return importlib.import_module(f'commands.{name}')
    except Exception:
        return None


def _declared(module, constant, function, *arguments):
    """The value of a declaration, _UNUSABLE, or _UNDECLARED when the module makes none."""
    if module is None:
        return _UNDECLARED
    declaration = getattr(module, function, None)
    if declaration is not None:
        if not callable(declaration):
            return _UNUSABLE
        try:
            return declaration(*[list(argument) if isinstance(argument, (list, tuple)) else argument
                                 for argument in arguments])
        except Exception:
            return _UNUSABLE
    return getattr(module, constant, _UNDECLARED)


def queue_kind(name, args, source='typed', module=None):
    """INSTANT for the command ``name`` when its module says it runs at once, else ORDERED.

    ``module`` is the command's module where the caller has it (the dispatch,
    which has just loaded it); without it the module is looked up by name.
    """
    declared = _declared(module if module is not None else command_module(name), 'QUEUE', 'queue_kind', args, source)
    return INSTANT if declared == INSTANT else ORDERED


def starts_model_request(name, args, module=None):
    """Whether the command ``name`` with ``args`` starts a request to a model, as its module declares."""
    declared = _declared(module if module is not None else command_module(name), 'STARTS_MODEL_REQUEST',
                         'starts_model_request', args)
    if declared is _UNDECLARED:
        return False
    return declared if isinstance(declared, bool) else True


def acts_on_jobs(name, args, module=None):
    """Whether the command ``name`` with ``args`` acts on the jobs that run, as its module declares."""
    declared = _declared(module if module is not None else command_module(name), 'ACTS_ON_JOBS',
                         'acts_on_jobs', args)
    return declared is True


def registers_console_button(name):
    """Whether the command ``name`` declares a Console button that comes back when a session loads.

    False for anything that is not a bare command name (a string that is an identifier and does not start with
    "_", which no command does), for a module that cannot be imported, and for a value that is not exactly True.
    """
    if not isinstance(name, str) or not name.isidentifier() or name.startswith('_'):
        return False
    return getattr(command_module(name), 'CONSOLE_BUTTON', False) is True


def _commands_that_mention(word):
    """The names of the commands whose source mentions ``word``, sorted, over the folders of the commands package.

    A cheap look before importing: only the modules that can declare something are imported, not every command
    (importing them all takes seconds). A name in an earlier folder hides the same name in a later one, as the
    import does, and a file that cannot be read is skipped.
    """
    try:
        folders = list(importlib.import_module('commands').__path__)
    except Exception:
        return []
    names = set()
    for folder in folders:
        for path in sorted(Path(folder).glob('*.py')):
            name = path.stem
            if name in names or name.startswith('_') or not name.isidentifier():
                continue
            try:
                if word in path.read_text(encoding='utf-8', errors='replace'):
                    names.add(name)
            except OSError:
                continue
    return sorted(names)


def session_file_pickers():
    """The Session card's pick buttons as the commands declare them: {file key: command name}.

    A command declares the file it picks with ``PICKS_SESSION_FILE = '<file key>'`` in its module, a constant that is
    a non-empty string; the key is the Viewer's name for the file (the card decides which keys it can change and
    what their buttons say). A module that cannot be imported, one without the declaration and one whose value is
    anything but such a string pick no file. When two commands declare the same file, neither is used: a button
    that could type either would run one by chance, so that file has no button, and a note on stderr names the
    commands. Read afresh at each call, so a plugin dropped into the commands folder is seen when the card next opens.
    """
    claims = {}
    for name in _commands_that_mention('PICKS_SESSION_FILE'):
        key = getattr(command_module(name), 'PICKS_SESSION_FILE', None)
        if isinstance(key, str) and key:
            claims.setdefault(key, []).append(name)
    pickers = {}
    for key, names in claims.items():
        if len(names) == 1:
            pickers[key] = names[0]
        else:
            print(f"The commands {', '.join(names)} all declare PICKS_SESSION_FILE = {key!r}, so the Session card "
                  "offers no button to pick that file.", file=sys.stderr)
    return pickers
