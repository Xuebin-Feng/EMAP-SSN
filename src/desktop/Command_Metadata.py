# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""What the command modules say about themselves, for the catalog and for the Agent, read without importing them.

The program never names a command here. A command is a module of src/commands, and
each module that wants to be described declares, as top-level assignments in one block
after its imports, up to four things:

* ``CATALOG_ENTRY``, the command's entry in the catalog the MCP server's
  get_command_catalog and the help command serve: a summary, its arguments (a
  finite set of words as choices, an open-ended one with none), the optional inputs it
  requires and whether it writes files or opens an interface of its own.

      CATALOG_ENTRY = entry('Summary.',
          argument('action', 'What it does.', choice('list'), choice('remove', 'delete')),
          requires=('alignment',), writes_files=True, opens_interface=True)

* ``AGENT_POLICY``, how a command of an Agent reply is handled: 'run' at once,
  'approve' only after the user presses Run in the Agent page, or 'refuse'; by default,
  and optionally by the lower-case first word after the command ('' for none).

      AGENT_POLICY = 'run'
      AGENT_POLICY = {'default': 'run', 'first_argument': {'cancel': 'approve'}}

* ``AGENT_PROMPT``, the text of the command's numbered item in the Agent's instructions
  (src/resources/agent/system_prompt.md has none): its usage line, then its bullets,
  without the number or the indentation, which are put on when the item is rendered; and
  ``AGENT_PROMPT_ORDER``, a number that puts the item in order (smaller first, ties by
  name, a command with none last). A command offered to the Agent that has no
  AGENT_PROMPT gets a short item made from its catalog summary and syntax, so a
  plugin dropped into src/commands reaches the Agent without anyone writing it one.

A module is never imported to read them. Its source is parsed and each declaration is
evaluated by a reader that accepts nothing but literals, ``+`` of two strings, calls
of entry, argument, choice and plain_choices, and the shared constants below
(EXPRESSION, CLUSTER_MODE, CLUSTER_PARAMETER, MIN_SIZE, HELP_DESCRIPTION). The module
imports the same names to run, so what Python evaluates and what is read are one value.
Reading a module's declarations therefore runs no handler and no code of the module,
this file imports no handler, and a plugin cannot compute its way to a policy: whoever
reviews a module sees every word the Viewer will go by.

What cannot be read is not obeyed, and the cautious value stands in for it: no
CATALOG_ENTRY (a declaration the reader cannot evaluate, or one of the wrong shape) is a
command that may write files and open an interface, with no arguments to name, listed as
metadata_declared: false; no AGENT_POLICY, or one that is invalid or assigned in any
other way than one plain top-level statement, is UNLISTED_POLICY: 'approve'. Never 'run'.
A policy word that can never match (not lower case, or holding a space) is invalid,
so a typo does not leave a form running at the default.

The commands that run in an Agent reply without review are pinned by a test
(tests/test_command_catalog.py, RUNS_WITHOUT_REVIEW): making a module 'run' fails it until
the list is changed on purpose, which is the one place to review.
"""
import ast
import math
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple


def choice(value, *aliases):
    return {'value': value, 'aliases': list(aliases)}


def plain_choices(*values):
    """A choice for each of values, none with an alias, to be spliced into argument(...) as *plain_choices(...)."""
    return [choice(value) for value in values]


def argument(name, description, *choices):
    return {'name': name, 'description': description, 'choices': list(choices)}


# Optional inputs a command cannot run without, as `requires`. A command that
# needs the embedding calls Command_Engine.require_embedding, and one that
# needs the alignment refuses to run without its rows.
REQUIRABLE_INPUTS = ('alignment', 'embedding')


def entry(summary, *arguments, requires=(), writes_files=False, opens_interface=False):
    """One command's catalog entry. writes_files: it can write files of its own (exports, saves, images,
    reports). opens_interface: it can open a dialog, page or window of its own, which needs a visible Viewer."""
    return {'summary': summary, 'arguments': list(arguments), 'requires': list(requires),
            'writes_files': writes_files, 'opens_interface': opens_interface}


EXPRESSION = ('Boolean node selection: quoted headers, #LABEL#, @file@, $sele$, '
              'residue positions, or {property comparison}; combine with &, |, !, ^. '
              'No spaces inside selection expressions. Referenced data must exist; '
              'negative residue positions and negative metadata range bounds use '
              'parentheses, e.g. K(-1) or {GRAVY=(-1)-0}.')
CLUSTER_MODE = argument('mode', 'Default leiden. Leiden requires graspologic-native; MCL requires markov_clustering, networkx and scipy. Leiden sorts the edges first, so its seeded result depends only on the network. MCL gives every node a self-loop as heavy as its strongest edge, so rescaling every score does not change its clusters, and runs on float32 weights in cluster and subcluster alike. Jaccard compares closed neighbourhoods (a node counts as its own neighbour), so an edge outside every triangle still scores above 0 and an isolated pair or clique edge scores 1.',
                        choice('leiden'), choice('mcl'), choice('jaccard'))
CLUSTER_PARAMETER = argument('parameter', 'Optional number: Leiden resolution defaults to 1.0; MCL inflation to 2.0, within 1.1 to 10.0; Jaccard threshold to 0.2.')
MIN_SIZE = argument('min_size', 'Optional integer, at least 1, default 10. Smaller subsets become noise. With 1, a node with no edge is a singleton cluster in every mode; with more, it is noise.')
# The help argument every catalog entry ends with, unless the module declares one of its own (export and label,
# which also take -?, do).
HELP_DESCRIPTION = 'Print command help instead of performing the operation.'


# --- How a command of an Agent reply is handled ---------------------------------------------------------------------

RUN, APPROVE, REFUSE = 'run', 'approve', 'refuse'
POLICY_KINDS = (RUN, APPROVE, REFUSE)


def policy(default, first_argument=None):
    """How a command in an Agent reply is handled, by its lower-case first
    argument ('' when there is none): 'run' at once, 'approve' only after the
    user presses Run in the Agent page, or 'refuse'."""
    return {'default': default, 'first_argument': dict(first_argument or {})}


# How a command whose module declares no valid AGENT_POLICY, such as a plugin dropped into src/commands, is handled
# in an Agent reply: it waits for approval, so a command added without a policy never runs unreviewed. 'refuse' is
# kept for what an agent must never run (it may not reconfigure itself).
UNLISTED_POLICY = policy('approve')


# --- Reading what a module declares ---------------------------------------------------------------------------------

COMMANDS_DIR = Path(__file__).resolve().parents[1] / 'commands'
DECLARATIONS = ('CATALOG_ENTRY', 'AGENT_POLICY', 'AGENT_PROMPT', 'AGENT_PROMPT_ORDER')

_FUNCTIONS = {'entry': entry, 'argument': argument, 'choice': choice, 'plain_choices': plain_choices}
_CONSTANTS = {'EXPRESSION': EXPRESSION, 'CLUSTER_MODE': CLUSTER_MODE, 'CLUSTER_PARAMETER': CLUSTER_PARAMETER,
              'MIN_SIZE': MIN_SIZE, 'HELP_DESCRIPTION': HELP_DESCRIPTION}


class Unreadable(ValueError):
    """A declaration that is more than literals, the shared constants and calls of the catalog's helpers."""


def _spliced(value):
    if not isinstance(value, (list, tuple)):
        raise Unreadable('only a list or tuple can be starred')
    return list(value)


def _evaluate(node):
    """The value of a declaration's expression, or Unreadable: literals, constants and the catalog's helpers only."""
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (str, int, float, bool)):
            return node.value
    elif isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return deepcopy(_CONSTANTS[node.id])
    elif isinstance(node, (ast.List, ast.Tuple)):
        items = []
        for element in node.elts:
            if isinstance(element, ast.Starred):
                items.extend(_spliced(_evaluate(element.value)))
            else:
                items.append(_evaluate(element))
        return items if isinstance(node, ast.List) else tuple(items)
    elif isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise Unreadable('a dictionary cannot be unpacked')
        return {_evaluate(key): _evaluate(value) for key, value in zip(node.keys, node.values)}
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _evaluate(node.operand)
        if isinstance(operand, (int, float)) and not isinstance(operand, bool):
            return -operand
    elif isinstance(node, ast.Call):
        function = node.func
        if isinstance(function, ast.Name) and function.id in _FUNCTIONS:
            positional = []
            for argument_node in node.args:
                if isinstance(argument_node, ast.Starred):
                    positional.extend(_spliced(_evaluate(argument_node.value)))
                else:
                    positional.append(_evaluate(argument_node))
            if any(keyword.arg is None for keyword in node.keywords):
                raise Unreadable('keyword arguments cannot be unpacked')
            keywords = {keyword.arg: _evaluate(keyword.value) for keyword in node.keywords}
            return _FUNCTIONS[function.id](*positional, **keywords)
    raise Unreadable(type(node).__name__)


def _value_nodes(tree):
    """{declaration name: the expression of its one plain top-level assignment, or None when it is made in any other way}.

    A declaration counts when the name is assigned by plain statements of the module's top level only; the last one
    wins, as when the module runs. A name that is also assigned or deleted anywhere else (inside an if, a try, a
    function, as one of several targets, augmented, imported, declared global) is not readable.
    """
    plain = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            plain[id(statement.targets[0])] = statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None and isinstance(statement.target, ast.Name):
            plain[id(statement.target)] = statement.value
    found, spoiled = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in DECLARATIONS and not isinstance(node.ctx, ast.Load):
            if id(node) in plain:
                found[node.id] = plain[id(node)]
            else:
                spoiled.add(node.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            spoiled.update(name for name in node.names if name in DECLARATIONS)
        elif isinstance(node, ast.arg) and node.arg in DECLARATIONS:
            spoiled.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            spoiled.update(alias.asname or alias.name for alias in node.names if (alias.asname or alias.name) in DECLARATIONS)
    return {name: (None if name in spoiled else found.get(name)) for name in set(found) | spoiled}


def _valid_entry(value):
    """The catalog entry that value says, in entry()'s form and order, or None when it is not one."""
    if not isinstance(value, dict) or not isinstance(value.get('summary'), str) or not value['summary'].strip():
        return None
    if not set(value) <= {'summary', 'arguments', 'requires', 'writes_files', 'opens_interface'}:
        return None
    arguments, requires = value.get('arguments', []), value.get('requires', [])
    writes_files, opens_interface = value.get('writes_files', False), value.get('opens_interface', False)
    if (not isinstance(arguments, (list, tuple)) or not isinstance(requires, (list, tuple))
            or not isinstance(writes_files, bool) or not isinstance(opens_interface, bool)
            or not all(isinstance(item, str) for item in requires)):
        return None
    for item in arguments:
        if (not isinstance(item, dict) or set(item) != {'name', 'description', 'choices'}
                or not isinstance(item['name'], str) or not item['name']
                or not isinstance(item['description'], str) or not isinstance(item['choices'], (list, tuple))):
            return None
        for option in item['choices']:
            if (not isinstance(option, dict) or set(option) != {'value', 'aliases'} or not isinstance(option['value'], str)
                    or not isinstance(option['aliases'], (list, tuple))
                    or not all(isinstance(alias, str) for alias in option['aliases'])):
                return None
    return entry(value['summary'], *[argument(item['name'], item['description'],
                                              *[choice(option['value'], *option['aliases']) for option in item['choices']])
                                     for item in arguments],
                 requires=requires, writes_files=writes_files, opens_interface=opens_interface)


def _valid_policy(value):
    """The policy that value says ('run' alone, or {'default': ..., 'first_argument': {...}}), or None when it is not one.

    Every word must be one the Viewer can look up: the first word after the command is lower-cased and is a single
    token, so a word with a capital or a space could never match and would leave its form at the default.
    """
    if isinstance(value, str):
        value = {'default': value}
    if not isinstance(value, dict) or not set(value) <= {'default', 'first_argument'}:
        return None
    default, words = value.get('default'), value.get('first_argument', {})
    if not isinstance(default, str) or default not in POLICY_KINDS or not isinstance(words, dict):
        return None
    for word, kind in words.items():
        if (not isinstance(word, str) or word != word.lower() or (word and word.split() != [word])
                or not isinstance(kind, str) or kind not in POLICY_KINDS):
            return None
    return policy(default, words)


def _valid_order(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _valid_prompt(value):
    return value if isinstance(value, str) and value.strip() else None


class Declarations(NamedTuple):
    """What one command module declares, each None when it declares none or what it declares cannot be used."""
    has_run: bool
    catalog_entry: dict | None
    agent_policy: dict | None
    agent_prompt: str | None
    agent_prompt_order: int | float | None


NO_DECLARATIONS = Declarations(False, None, None, None, None)


def read_declarations(tree):
    """The declarations of the module whose source is ``tree`` (an ast.Module), without running any of it."""
    nodes = _value_nodes(tree)

    def read(name, valid):
        node = nodes.get(name)
        if node is None:
            return None
        try:
            return valid(_evaluate(node))
        except Exception:
            return None

    has_run = any(isinstance(statement, ast.FunctionDef) and statement.name == 'run' for statement in tree.body)
    return Declarations(has_run, read('CATALOG_ENTRY', _valid_entry), read('AGENT_POLICY', _valid_policy),
                        read('AGENT_PROMPT', _valid_prompt), read('AGENT_PROMPT_ORDER', _valid_order))


# What reading one file gave, by the file's path, valid while its modification time and size are the same: the Agent
# page asks for the policy of every command it runs and for the whole prompt on every turn.
_read_files = {}


def declarations_of(path):
    """The declarations of the module at ``path``; NO_DECLARATIONS for a file that is missing or cannot be parsed."""
    path = Path(path)
    try:
        status = path.stat()
    except OSError:
        return NO_DECLARATIONS
    signature = (status.st_mtime_ns, status.st_size)
    cached = _read_files.get(str(path))
    if cached is not None and cached[0] == signature:
        return cached[1]
    try:
        declared = read_declarations(ast.parse(path.read_text(encoding='utf-8'), filename=str(path)))
    except (OSError, SyntaxError, ValueError, RecursionError):
        declared = NO_DECLARATIONS
    _read_files[str(path)] = (signature, declared)
    return declared


# --- Where the commands are ---------------------------------------------------------------------------------------

def command_directories():
    """The folders a command's module is searched for, in the order the dispatch searches them.

    That is the commands package's own path where the program has loaded it (the VR viewer's folder comes first
    there, a test's scratch folder last), otherwise the main program's folder of commands.
    """
    package = sys.modules.get('commands')
    directories = [Path(directory) for directory in getattr(package, '__path__', ())]
    return directories or [COMMANDS_DIR]


def _valid_name(name):
    return isinstance(name, str) and name.isidentifier() and not name.startswith('_')


def command_paths(directories=None):
    """[(name, path)] of every module in the folders, by name; where two folders have one, the first folder's."""
    found = {}
    for directory in (command_directories() if directories is None else directories):
        try:
            candidates = sorted(Path(directory).glob('*.py'))
        except OSError:
            continue
        for path in candidates:
            if _valid_name(path.stem):
                found.setdefault(path.stem, path)
    return sorted(found.items())


def declarations(name, directories=None):
    """The declarations of command ``name`` (the first module of that name in the folders), NO_DECLARATIONS if
    there is none or it has no run(): what is not a command declares nothing."""
    if not _valid_name(name):
        return NO_DECLARATIONS
    for directory in (command_directories() if directories is None else directories):
        path = Path(directory) / f'{name}.py'
        if path.is_file():
            declared = declarations_of(path)
            return declared if declared.has_run else NO_DECLARATIONS
    return NO_DECLARATIONS


# --- The catalog -----------------------------------------------------------------------------------------------------

def catalog_entry(declared):
    """The catalog entry the declarations say: the declared entry with the help argument added unless it declares one,
    and the policy the module declares (UNLISTED_POLICY if none). ``declared`` must have a catalog_entry."""
    metadata = deepcopy(declared.catalog_entry)
    if not any(item['name'] == 'help' for item in metadata['arguments']):
        metadata['arguments'].append(argument('help', HELP_DESCRIPTION, choice('help', '-h', '--help')))
    metadata['agent_policy'] = deepcopy(declared.agent_policy or UNLISTED_POLICY)
    return metadata


class _Declared(Mapping):
    """A read-only view of one declaration of every command: {command name: value}, read from the modules when asked.

    It stands for the tables that used to be written here, for readers that want one (the VR viewer's help and the
    tests). Nothing is stored in it, so a command added or changed is there at once. A name is a key when its module
    has a run() and declares a value that can be used.
    """

    def __init__(self, directories, attribute, build):
        self._directories, self._attribute, self._build = directories, attribute, build

    def _declared(self, name):
        declared = declarations(name, self._directories())
        return declared if getattr(declared, self._attribute) is not None else None

    def __getitem__(self, name):
        declared = self._declared(name)
        if declared is None:
            raise KeyError(name)
        return self._build(declared)

    def __iter__(self):
        for name, path in command_paths(self._directories()):
            declared = declarations_of(path)
            if declared.has_run and getattr(declared, self._attribute) is not None:
                yield name

    def __len__(self):
        return sum(1 for _ in self)

    def __repr__(self):
        return f'<{self._attribute} of the commands: {", ".join(self)}>'


# The main program's own commands, as the VR viewer's help reads them: its folder of commands comes first on the
# commands package's path and holds modules that declare nothing, which must not hide the main program's entries.
COMMAND_METADATA = _Declared(lambda: [COMMANDS_DIR], 'catalog_entry', catalog_entry)
# How a command of an Agent reply is handled, for every command whose module declares it, in the order of the
# dispatch's search.
AGENT_POLICY = _Declared(lambda: command_directories(), 'agent_policy', lambda declared: deepcopy(declared.agent_policy))


def get_command_metadata(name):
    """Return independent response data so callers cannot mutate the catalog."""
    return COMMAND_METADATA[name]


def default_command_metadata(summary, agent_policy_of=None):
    """The entry of a command module that has none above: nothing is declared, so be cautious.

    It says it may write files and open an interface, takes no arguments it could name, requires nothing it
    could know of, and in an Agent reply is handled as its module's policy says (waits for approval if it
    declares none, as agent_policy treats any command without one).
    """
    return {'summary': summary, 'arguments': [], 'requires': [], 'writes_files': True, 'opens_interface': True,
            'agent_policy': deepcopy(UNLISTED_POLICY if agent_policy_of is None else agent_policy_of)}


# --- The Agent -----------------------------------------------------------------------------------------------------

def _rule(command):
    """The policy that applies to the command's module: its declared one, else UNLISTED_POLICY."""
    return declarations(command).agent_policy or UNLISTED_POLICY


def agent_policy(command):
    """'run', 'approve' or 'refuse' for one command line in an Agent reply.

    The command name is read the way Command_Engine dispatches it. A command
    whose module declares no usable AGENT_POLICY, or that is no command at all,
    waits for approval, so a command added without a policy never runs unreviewed.
    """
    words = command.split()
    rule = _rule(words[0].lower() if words else '')
    return rule['first_argument'].get(words[1].lower() if len(words) > 1 else '', rule['default'])


def agent_policy_summary():
    """AGENT_POLICY grouped the way the Agent's instructions list it.

    Returns (plain, by_form). plain maps each class, 'run', 'approve' and
    'refuse', to the names of the commands handled that way whatever follows
    them, by command name. by_form lists the others, each as
    (name, default class, {first argument: class}), with '' standing for no
    argument at all: the first word after the command decides how they are
    handled. The Agent's system prompt is written from this, so it cannot
    disagree with agent_policy(). A command that declares no policy is in
    neither: it is the one the prompt says waits for approval.
    """
    plain = {kind: [] for kind in POLICY_KINDS}
    by_form = []
    for name in AGENT_POLICY:
        rule = declarations(name).agent_policy
        exceptions = {word: kind for word, kind in rule['first_argument'].items() if kind != rule['default']}
        if exceptions:
            by_form.append((name, rule['default'], exceptions))
        else:
            plain[rule['default']].append(name)
    return plain, by_form


def unlisted_agent_policy():
    """How an Agent reply's command is handled when its module declares no policy: UNLISTED_POLICY's default."""
    return UNLISTED_POLICY['default']


def refused_in_every_form(rule):
    """Whether a policy refuses the command whatever follows it."""
    return rule['default'] == REFUSE and all(kind == REFUSE for kind in rule['first_argument'].values())


class PromptItem(NamedTuple):
    """One command's place in the Agent's instructions: its declared text (None: make one) and its order (None: last)."""
    name: str
    path: Path
    text: str | None
    order: int | float | None


def agent_prompt_items():
    """The commands the Agent's instructions describe, in order: those a reply may run in some form.

    A command refused in every form is not offered, whatever its module declares. The order is the commands'
    AGENT_PROMPT_ORDER, smaller first, then by name, with a command that declares none last.
    """
    items = []
    for name, path in command_paths():
        declared = declarations_of(path)
        if declared.has_run and not refused_in_every_form(declared.agent_policy or UNLISTED_POLICY):
            items.append(PromptItem(name, path, declared.agent_prompt, declared.agent_prompt_order))
    return sorted(items, key=lambda item: (item.order is None, item.order or 0, item.name))
