# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""What a command module declares about how the Viewer dispatches it.

A command is a module of src/commands, with run(viewer, args). The Viewer's
dispatch (Command_Engine), its command queue (Command_Queue) and the command
portal that serves MCP and the Agent page (Viewer_Command_Portal) never name a
command. Where one has to be treated differently from the rest, they ask its
module, and a module that says nothing is treated cautiously. A new command, or
a plugin dropped into src/commands, is therefore dispatched correctly without
any change to the program.

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

A function wins over the constant of the same declaration. A value that is not
what the declaration asks for, or a function that raises, counts as undeclared
(ordered; not acting on jobs), except for STARTS_MODEL_REQUEST: a declaration
that cannot be read counts as true there, so that a mistake refuses the command
rather than lets it start a request. A module that cannot be imported has
declared nothing, and its command fails when it is run, as it did before.
"""
import importlib

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
