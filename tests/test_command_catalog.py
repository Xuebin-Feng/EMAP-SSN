"""Static Viewer command catalog (desktop.Command_Metadata, desktop.Viewer_Inspection.command_catalog).

Covers the catalog's argument metadata and finite choices, which must be literal
tokens of each command's parser, the syntax and help text it extracts from the
command sources without importing or executing any handler, how a command module
declares what the catalog and the Agent say about it (CATALOG_ENTRY, AGENT_POLICY,
AGENT_PROMPT) and how those are read, and the policy for commands in Agent replies:
what each shipped command and form is handled as, and which run without review.
"""
import ast
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

COMMANDS_DIR = Path(__file__).resolve().parents[1] / 'src' / 'commands'
sys.path.insert(0, str(COMMANDS_DIR.parent))
from desktop import Command_Metadata  # noqa: E402
from desktop.Command_Metadata import (  # noqa: E402
    AGENT_POLICY, COMMAND_METADATA, DECLARATIONS, UNLISTED_POLICY, agent_policy, agent_policy_summary,
    agent_prompt_items, declarations_of, default_command_metadata, entry, get_command_metadata, read_declarations)
from desktop.Viewer_Inspection import command_catalog, _command_syntax  # noqa: E402

# The commands that ship in src/commands. A test that names them checks what is known of them; none is a list that
# a command added later has to join.
SHIPPED = ('agent', 'alignment', 'cluster', 'color', 'console', 'embedding', 'esmfold', 'export', 'group', 'help', 'hide',
           'jobs', 'label', 'logo', 'meta', 'offset', 'print', 'query', 'redo', 'reference', 'reset', 'run', 'save',
           'select', 'spectrum', 'subcluster', 'undo', 'zoom')


def string_literals(command):
    """The string constants of a command's source other than its declarations: what its parser can match."""
    tree = ast.parse((COMMANDS_DIR / (command + '.py')).read_text(encoding='utf-8'))
    declared = {id(node) for statement in tree.body if isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id in DECLARATIONS for target in statement.targets)
                for node in ast.walk(statement)}
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in declared}


class MetadataTests(unittest.TestCase):
    def test_complete_independent_metadata_and_finite_choice_sources(self):
        catalog = command_catalog()['commands']
        # Every command that declares a catalog entry is catalogued with it; one that declares none is marked so.
        self.assertEqual(set(COMMAND_METADATA), {c['command'] for c in catalog if 'metadata_declared' not in c})
        self.assertTrue(set(SHIPPED) <= set(COMMAND_METADATA), set(SHIPPED) - set(COMMAND_METADATA))
        for entry in catalog:
            name = entry['command']
            with self.subTest(command=name):
                self.assertTrue(entry['summary'])
                self.assertIsNone(entry['help'])
                self.assertEqual(len({a['name'] for a in entry['arguments']}), len(entry['arguments']))
                literals = string_literals(name)
                for arg in entry['arguments']:
                    self.assertEqual(set(arg), {'name', 'description', 'choices'})
                    self.assertTrue(arg['description'])
                    tokens = []
                    for item in arg['choices']:
                        self.assertEqual(set(item), {'value', 'aliases'})
                        self.assertIsInstance(item['aliases'], list)
                        tokens += [item['value'], *item['aliases']]
                    self.assertEqual(len(tokens), len(set(tokens)))
                    # Finite choices must be actual parser literals, not extracted
                    # from prose help. Reset plural normalization is tested below.
                    if name != 'reset':
                        self.assertTrue(set(tokens) <= literals, (name, arg['name'], set(tokens) - literals))
        copy = get_command_metadata('reset')
        copy['arguments'][0]['choices'][0]['aliases'].append('not-a-target')
        self.assertNotIn('not-a-target', repr(command_catalog('reset')))

    def test_discovery_does_not_import_handlers(self):
        with mock.patch('importlib.import_module', side_effect=AssertionError('handler imported')):
            self.assertTrue(command_catalog()['commands'])


class CatalogTests(unittest.TestCase):
    def test_usage_sections_do_not_collect_examples_or_prose(self):
        text = '''Usage: reset colors
                  reset sizes
                  reset colors

Description:
  reset is reversible
Examples:
  reset shapes
Usage:
reset hide
'''
        self.assertEqual(_command_syntax(text, 'reset'),
                         ['reset colors', 'reset sizes', 'reset hide'])
        self.assertEqual(_command_syntax('No usage available.', 'reset'), [])
        self.assertEqual(_command_syntax('Usage:\n  <target>', 'reset'), [])

    def test_catalog_is_read_only_and_preserves_detailed_help(self):
        with mock.patch('Command_Engine.execute_command', side_effect=AssertionError('executed')):
            general = command_catalog()
            detailed = command_catalog('reset')['commands'][0]
        self.assertTrue(set(SHIPPED) <= {c['command'] for c in general['commands']})
        self.assertTrue(all(c['help'] is None and isinstance(c['syntax'], list)
                            for c in general['commands']))
        self.assertIn('reset <TARGET_1> [TARGET_2] ...', detailed['syntax'])
        for target in ('colors', 'sizes', 'shapes', 'clusters', 'groups', 'hide', 'network', 'order, layer'):
            self.assertIn(target, detailed['help'])
        self.assertIn('arguments={"command":"reset"}', general['note'])
        with self.assertRaisesRegex(ValueError, 'Unknown command'):
            command_catalog('missing')

    def test_help_built_with_an_f_string_is_read_whole(self):
        """meta's help is an f-string; the catalog used to stop at its first placeholder."""
        meta = command_catalog('meta')['commands'][0]
        for form in ('meta download <filename>', 'meta show/display <property_name>',
                     'meta delete/remove/clear <property_name> [property_name ...]'):
            self.assertIn(form, meta['syntax'])
        self.assertIn('Deleting every column with "all" is not supported.', meta['help'])
        # A placeholder reads as <name>: braces would mean a metadata predicate.
        self.assertIn('the metadata directory: <meta_dir>', meta['help'])
        # f-strings in run() are runtime message templates, not syntax.
        self.assertNotIn('meta <first_arg> <property_name> [property_name ...]', meta['syntax'])

    def test_an_alternative_usage_form_is_catalogued(self):
        self.assertEqual(
            _command_syntax('Usage: label [A]\n   or: label [B]\nNotes:\n  or: label [C]', 'label'),
            ['label [A]', 'label [B]'],
        )
        self.assertIn('label [TARGET] [key value] [<key 2> <value 2> ...] [NAME]',
                      command_catalog('label')['commands'][0]['syntax'])


def calls_function(command, name):
    """Whether the command's source calls NAME, as a bare name or an attribute."""
    tree = ast.parse((COMMANDS_DIR / (command + '.py')).read_text(encoding='utf-8'))
    return any(
        isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
        for node in ast.walk(tree)
    )


def tests_for_missing_alignment_rows(command):
    """Whether the command's source refuses to run when the alignment has no rows (alignment.aln is None)."""
    tree = ast.parse((COMMANDS_DIR / (command + '.py')).read_text(encoding='utf-8'))
    return any(
        isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute) and node.left.attr == 'aln'
        and any(isinstance(op, ast.Is) for op in node.ops)
        and any(isinstance(value, ast.Constant) and value.value is None for value in node.comparators)
        for node in ast.walk(tree)
    )


class RequiredInputTests(unittest.TestCase):
    """`requires` names the optional inputs a command cannot run without."""

    def test_requires_lists_only_known_inputs_and_reaches_the_catalog(self):
        from desktop.Command_Metadata import REQUIRABLE_INPUTS
        for entry in command_catalog()['commands']:
            with self.subTest(command=entry['command']):
                self.assertIsInstance(entry['requires'], list)
                self.assertTrue(set(entry['requires']) <= set(REQUIRABLE_INPUTS), entry['requires'])
                self.assertEqual(len(entry['requires']), len(set(entry['requires'])))
        self.assertEqual(
            {name for name, metadata in COMMAND_METADATA.items() if 'alignment' in metadata['requires'] and name in SHIPPED},
            {'label', 'logo', 'query'},
        )

    def test_the_embedding_is_required_exactly_where_require_embedding_is_called(self):
        calling = {name for name in COMMAND_METADATA if calls_function(name, 'require_embedding')}
        declaring = {name for name, metadata in COMMAND_METADATA.items() if 'embedding' in metadata['requires']}
        self.assertEqual(calling, declaring)

    def test_a_command_requiring_the_alignment_refuses_to_run_without_its_rows(self):
        requiring = [name for name, metadata in COMMAND_METADATA.items() if 'alignment' in metadata['requires']]
        self.assertTrue(requiring)
        for name in requiring:
            with self.subTest(command=name):
                self.assertTrue(tests_for_missing_alignment_rows(name))

    def test_the_checks_see_a_call_and_a_test_in_any_form(self):
        sources = {
            'bare': 'def run(viewer, args):\n    embedding = require_embedding(viewer)\n',
            'attribute': 'def run(viewer, args):\n    embedding = Command_Engine.require_embedding(viewer)\n',
            'mention': 'def run(viewer, args):\n    """Unlike require_embedding, needs nothing."""\n',
            'rows': 'def run(viewer, args):\n    if viewer.alignment.aln is None:\n        return\n',
        }
        with mock.patch.object(Path, 'read_text', autospec=True,
                               side_effect=lambda path, encoding=None: sources[path.stem]):
            self.assertTrue(calls_function('bare', 'require_embedding'))
            self.assertTrue(calls_function('attribute', 'require_embedding'))
            self.assertFalse(calls_function('mention', 'require_embedding'))
            self.assertTrue(tests_for_missing_alignment_rows('rows'))
            self.assertFalse(tests_for_missing_alignment_rows('bare'))


PLUGINS = {
    # A docstring says what it is; nothing else is declared.
    'zdoc': ('''"""Scratch command with a docstring.

Second paragraph, never used.
"""
def run(viewer, args):
    return None
'''),
    # No docstring: the help's Description line, then its Usage line.
    'zhelp': ('''def print_help():
    print("Usage: zhelp <thing> [more]\\n  or: zhelp reset\\nDescription: Does a scratch thing.")


def run(viewer, args):
    print_help()
'''),
    'zusage': ('''def print_help():
    print("Usage: zusage <thing>")


def run(viewer, args):
    print_help()
'''),
    # Nothing at all but the entry point.
    'zbare': "def run(viewer, args):\n    return None\n",
    # A planned command whose plan is heavy: it computes on the command queue's worker thread.
    'zheavy': ('''from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    heavy = True


@dataclass(frozen=True)
class QuickPlan:
    heavy = False


def plan(viewer, args):
    return Plan()


def run(viewer, args):
    return None
'''),
    # Only light plans: nothing goes on after the command returns.
    'zlight': ('''class Plan:
    heavy = False


def plan(viewer, args):
    return Plan()


def run(viewer, args):
    return None
'''),
    # A file job for the background scheduler, a tracked script, and a worker process.
    'zfile': "def run(viewer, args):\n    viewer.background_job_scheduler.enqueue(command_name='zfile')\n",
    'zscript': "def run(viewer, args):\n    from Viewer_Worker_Tracking import ScriptTracker\n    ScriptTracker(None, 'x.py')\n",
    'zworker': "import subprocess\n\n\ndef run(viewer, args):\n    subprocess.Popen(['x'])\n",
    'ztyped': "def run(viewer, args):\n    from Viewer_Worker_Tracking import TypedScriptRun\n    TypedScriptRun(viewer, 'x.py', None)\n",
    'ztracked': "def run(viewer, args):\n    from Viewer_Worker_Tracking import track_worker\n    track_worker(None)\n",
    # Mentions the names without calling them: not background work.
    'zmention': ("def run(viewer, args):\n    \"\"\"Unlike enqueue, Popen and ScriptTracker, runs now; heavy = True is prose.\"\"\"\n"
                 "    starters = [enqueue, Popen, ScriptTracker]\n    heavy_plan = heavy\n    return starters, heavy_plan\n"),
    # Not a command: no run().
    'zhelper': "def helper():\n    return 1\n",
    '_private': "def run(viewer, args):\n    return None\n",
}


class PluginWithoutMetadataTests(unittest.TestCase):
    """A command module dropped into src/commands needs no entry in COMMAND_METADATA to be catalogued."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.commands = Path(cls.folder.name) / 'commands'
        shutil.copytree(COMMANDS_DIR, cls.commands, ignore=shutil.ignore_patterns('__pycache__'))
        for name, source in PLUGINS.items():
            (cls.commands / (name + '.py')).write_text(source, encoding='utf-8')
        cls.catalog = command_catalog(commands_dir=cls.commands)
        cls.entries = {entry['command']: entry for entry in cls.catalog['commands']}
        cls.shipped = {entry['command']: entry for entry in command_catalog()['commands']}

    def test_every_command_is_still_listed_and_no_other_entry_changes(self):
        self.assertTrue(set(SHIPPED) <= set(self.shipped))
        for name, expected in self.shipped.items():
            with self.subTest(command=name):
                self.assertEqual(self.entries[name], expected)
                self.assertEqual(list(self.entries[name]), list(expected))
        for name in self.shipped:
            detailed = command_catalog(name, commands_dir=self.commands)['commands']
            self.assertEqual(detailed, command_catalog(name)['commands'])

    def test_only_modules_with_a_run_entry_point_become_commands(self):
        added = set(self.entries) - set(self.shipped)
        self.assertEqual(added, {'zdoc', 'zhelp', 'zusage', 'zbare', 'zheavy', 'zlight', 'zfile', 'zscript', 'zworker',
                                 'ztyped', 'ztracked', 'zmention'})
        self.assertEqual(self.catalog['note'], command_catalog()['note'])

    def test_the_summary_comes_from_the_docstring_then_the_help_then_a_plain_sentence(self):
        self.assertEqual(self.entries['zdoc']['summary'], 'Scratch command with a docstring.')
        self.assertEqual(self.entries['zhelp']['summary'], 'Does a scratch thing.')
        self.assertEqual(self.entries['zusage']['summary'], 'zusage <thing>')
        self.assertEqual(self.entries['zbare']['summary'], 'The zbare command; its module declares no description.')

    def test_nothing_is_claimed_beyond_what_the_module_shows(self):
        for name in sorted(set(self.entries) - set(self.shipped)):
            with self.subTest(command=name):
                found = self.entries[name]
                self.assertIs(found['metadata_declared'], False)
                self.assertEqual((found['arguments'], found['requires']), ([], []))
                # Cautious about what it might do, since nothing says.
                self.assertTrue(found['writes_files'] and found['opens_interface'])
                self.assertEqual(found['agent_policy'], {'default': 'approve', 'first_argument': {}})
                self.assertTrue(found['summary'])
                self.assertIsNone(found['help'])
                self.assertEqual(list(found)[:10], ['command', 'summary', 'arguments', 'requires', 'agent_policy',
                                                    'writes_files', 'opens_interface', 'background_work', 'syntax', 'help'])
        for name in self.shipped:
            self.assertNotIn('metadata_declared', self.entries[name])

    def test_syntax_and_help_are_read_from_the_module_as_for_any_command(self):
        self.assertEqual(self.entries['zhelp']['syntax'], ['zhelp <thing> [more]', 'zhelp reset'])
        self.assertEqual(self.entries['zbare']['syntax'], [])
        detailed = command_catalog('zhelp', commands_dir=self.commands)['commands'][0]
        self.assertIn('Description: Does a scratch thing.', detailed['help'])
        self.assertIs(detailed['metadata_declared'], False)

    def test_background_work_is_read_from_what_the_module_starts(self):
        expected = {'zheavy': True, 'zfile': True, 'zscript': True, 'zworker': True, 'ztyped': True, 'ztracked': True,
                    'zlight': False, 'zbare': False, 'zdoc': False, 'zhelp': False, 'zmention': False}
        for name, background in expected.items():
            with self.subTest(command=name):
                self.assertIs(self.entries[name]['background_work'], background)

    def test_the_shipped_commands_that_queue_or_compute_in_the_background_say_so(self):
        for name in ('print', 'cluster', 'subcluster', 'label', 'logo', 'esmfold', 'run'):
            with self.subTest(command=name):
                self.assertIs(self.shipped[name]['background_work'], True)
        for name in ('select', 'zoom', 'jobs', 'color', 'save', 'export', 'meta', 'alignment', 'embedding', 'agent'):
            with self.subTest(command=name):
                self.assertIs(self.shipped[name]['background_work'], False)

    def test_the_default_policy_is_the_one_agent_replies_already_apply_to_unknown_commands(self):
        self.assertEqual(UNLISTED_POLICY, {'default': 'approve', 'first_argument': {}})
        for name in ('zbare', 'zunknown'):
            self.assertEqual(agent_policy(name), 'approve')
            self.assertEqual(agent_policy(name + ' anything'), 'approve')
        self.assertEqual(self.entries['zbare']['agent_policy']['default'], agent_policy('zbare'))
        # No entry ever runs unreviewed or is refused for lack of a policy; and the shared constant is never handed out.
        self.assertNotIn(UNLISTED_POLICY['default'], ('run', 'refuse'))
        self.entries['zbare']['agent_policy']['default'] = 'run'
        self.assertEqual(default_command_metadata('x')['agent_policy'], {'default': 'approve', 'first_argument': {}})
        self.assertEqual(UNLISTED_POLICY['default'], 'approve')

    def test_the_flags_of_the_shipped_commands_are_declared_in_their_entries(self):
        self.assertEqual({n for n, m in COMMAND_METADATA.items() if m['writes_files'] and n in SHIPPED},
                         {'esmfold', 'export', 'label', 'logo', 'meta', 'print', 'run', 'save', 'select'})
        self.assertEqual({n for n, m in COMMAND_METADATA.items() if m['opens_interface'] and n in SHIPPED},
                         {'agent', 'alignment', 'embedding', 'esmfold', 'export', 'label', 'logo', 'meta', 'print', 'run'})
        self.assertEqual(entry('x')['writes_files'], False)
        self.assertEqual(entry('x')['opens_interface'], False)
        for name, metadata in COMMAND_METADATA.items():
            self.assertEqual(self.shipped[name]['writes_files'], metadata['writes_files'])
            self.assertEqual(self.shipped[name]['opens_interface'], metadata['opens_interface'])

    def test_the_catalog_still_reaches_agents_through_the_portal_and_without_a_viewer(self):
        from desktop.Viewer_Inspection import ViewerInspectionService
        service = ViewerInspectionService(mock.Mock())
        general = service.command_action('get_command_catalog', {})
        self.assertEqual({e['command'] for e in general['commands']}, set(self.shipped))
        detailed = service.command_action('get_command_catalog', {'command': 'print'})
        self.assertIs(detailed['commands'][0]['background_work'], True)


# What every shipped command, and every form of it, is handled as in an Agent reply. It is what the one central table
# said before each module declared its own policy: the default, then the first words handled otherwise ('' is no
# argument at all). The modules must declare exactly this, in this order of words.
TODAYS_POLICY = {
    'agent': ('refuse', {}),
    'alignment': ('approve', {}),
    'cluster': ('run', {}),
    'color': ('run', {}),
    'console': ('refuse', {}),
    'embedding': ('approve', {}),
    'esmfold': ('approve', {}),
    'export': ('approve', {}),
    'group': ('run', {}),
    'help': ('refuse', {}),
    'hide': ('run', {}),
    'jobs': ('run', {'cancel': 'approve'}),
    'label': ('approve', {'reset': 'run'}),
    'logo': ('approve', {}),
    'meta': ('approve', dict.fromkeys(
        ('', 'show', 'display', 'delete', 'remove', 'clear', 'help', '-h', '--help', '--register-only'), 'run')),
    'offset': ('run', {}),
    'print': ('approve', {}),
    'query': ('run', {}),
    'redo': ('run', {}),
    'reference': ('run', {}),
    'reset': ('run', {}),
    'run': ('approve', {}),
    'save': ('approve', {}),
    'select': ('run', {'save': 'approve'}),
    'spectrum': ('run', {}),
    'subcluster': ('run', {}),
    'undo': ('run', {}),
    'zoom': ('run', {}),
}

# THE REVIEW POINT. Every command and form of src/commands that an Agent reply runs at once, with no review by the
# user. A command is listed by name; '*' is every form of it that its default covers, a word is the first word after
# the command, and '' no word at all. A module that declares AGENT_POLICY = 'run' (or a form that runs) fails
# test_only_the_pinned_commands_and_forms_run_without_review until it is added here, which is the moment to ask
# whether an Agent reply should be allowed to do that unreviewed: the command must change only the view or the
# session's in-memory state, never write or import a file, start a process, open a dialog or spend credit.
RUNS_WITHOUT_REVIEW = {
    'cluster': {'*'},
    'color': {'*'},
    'group': {'*'},
    'hide': {'*'},
    'jobs': {'*'},
    'label': {'reset'},
    'meta': {'', 'show', 'display', 'delete', 'remove', 'clear', 'help', '-h', '--help', '--register-only'},
    'offset': {'*'},
    'query': {'*'},
    'redo': {'*'},
    'reference': {'*'},
    'reset': {'*'},
    'select': {'*'},
    'spectrum': {'*'},
    'subcluster': {'*'},
    'undo': {'*'},
    'zoom': {'*'},
}


def unreviewed():
    """{command: forms} of every command found by the dispatch's search that an Agent reply runs without review.

    Read through the public functions only, as the Viewer does: the command's default, and each first word its
    policy names.
    """
    found = {}
    for name in AGENT_POLICY:
        rule = AGENT_POLICY[name]
        forms = {word for word, kind in rule['first_argument'].items() if agent_policy(f'{name} {word}'.strip()) == 'run'}
        if rule['default'] == 'run':
            forms.add('*')
        if forms:
            found[name] = forms
    return found


class AgentPolicyTests(unittest.TestCase):
    """How a command in an Agent reply is handled; Viewer_Command_Portal applies it."""

    def test_every_shipped_command_and_form_is_handled_as_it_was_before_the_modules_declared_it(self):
        self.assertEqual(sorted(TODAYS_POLICY), sorted(SHIPPED))
        for name, (default, forms) in TODAYS_POLICY.items():
            with self.subTest(command=name):
                self.assertEqual(AGENT_POLICY[name], {'default': default, 'first_argument': forms})
                # The order of the words is the order the Agent's instructions list them in.
                self.assertEqual(list(AGENT_POLICY[name]['first_argument']), list(forms))
                for word in ['', *forms, 'zz-unlisted', 'HELP', 'Cancel', 'Save', 'SHOW', '--x', '-?']:
                    expected = forms.get(word.lower(), default)
                    texts = ([name, name.upper(), f'  {name.title()}  '] if not word else
                             [f'{name} {word}', f'{name.upper()} {word.upper()}', f'{name}  {word} 1 "two" three',
                              f'  {name.title()} {word}  '])
                    for text in texts:
                        self.assertEqual(agent_policy(text), expected, text)

    def test_the_summary_groups_the_shipped_policy_as_the_table_did(self):
        plain, by_form = agent_policy_summary()
        for kind in ('run', 'approve', 'refuse'):
            self.assertEqual([name for name in plain[kind] if name in SHIPPED],
                             [name for name, (default, forms) in TODAYS_POLICY.items()
                              if default == kind and not any(other != kind for other in forms.values())])
        self.assertEqual([entry for entry in by_form if entry[0] in SHIPPED],
                         [(name, default, {word: kind for word, kind in forms.items() if kind != default})
                          for name, (default, forms) in TODAYS_POLICY.items()
                          if any(kind != default for kind in forms.values())])

    def test_every_command_has_a_policy_in_the_catalog(self):
        self.assertTrue(set(SHIPPED) <= set(AGENT_POLICY), set(SHIPPED) - set(AGENT_POLICY))
        for entry in command_catalog()['commands']:
            with self.subTest(command=entry['command']):
                rule = entry['agent_policy']
                self.assertEqual(rule, AGENT_POLICY.get(entry['command'], UNLISTED_POLICY))
                self.assertTrue({rule['default'], *rule['first_argument'].values()} <= {'run', 'approve', 'refuse'})
                # Keywords are the command's own parser literals.
                self.assertTrue(set(rule['first_argument']) - {''} <= string_literals(entry['command']))

    def test_commands_that_write_run_or_spend_wait_and_agent_is_refused(self):
        cases = {
            'run': ['select "a"', 'select invert', 'color "a" red', 'zoom 10', 'hide single', 'cluster',
                    'subcluster cluster_1', 'group "a" g1', 'spectrum {GRAVY}', 'query [1-3]', 'reset colors',
                    'undo', 'redo', 'offset 2', 'reference P12', 'label reset', 'meta', 'meta show GRAVY', 'jobs',
                    'meta delete GRAVY', 'meta HELP'],
            'approve': ['select save picked.txt', 'SELECT Save picked.fasta', 'save', 'save x.h5', 'print svg',
                        'export clusters', 'logo [1-3]', 'label', 'label clusters 40 98', 'esmfold',
                        'esmfold large multi', 'run', 'alignment msa.fasta', 'embedding emb.h5', 'meta upload a.csv',
                        'meta download', 'meta table.csv', 'unknown_command', 'jobs cancel', 'JOBS cancel all'],
            'refuse': ['agent', 'agent off', 'agent <Local>', 'agent "hello"', 'AGENT help'],
        }
        for expected, commands in cases.items():
            for command in commands:
                with self.subTest(command=command):
                    self.assertEqual(agent_policy(command), expected)

    def test_only_the_pinned_commands_and_forms_run_without_review(self):
        self.assertEqual(
            unreviewed(), RUNS_WITHOUT_REVIEW,
            'The commands an Agent reply runs without the user\'s review changed. A module that now declares '
            "AGENT_POLICY = 'run' (or a form that runs) must be added to RUNS_WITHOUT_REVIEW on purpose, after "
            'asking whether an Agent reply may do that unreviewed; one that no longer does is removed.')

    def test_the_pin_sees_a_command_a_module_makes_run_and_one_it_stops_running(self):
        with tempfile.TemporaryDirectory() as folder:
            commands = Path(folder) / 'commands'
            shutil.copytree(COMMANDS_DIR, commands, ignore=shutil.ignore_patterns('__pycache__'))
            (commands / 'zrunner.py').write_text("AGENT_POLICY = 'run'\n\n\ndef run(viewer, args):\n    return None\n", encoding='utf-8')
            for name, old, new in (('save', "AGENT_POLICY = 'approve'", "AGENT_POLICY = 'run'"),
                                   ('select', "{'save': 'approve'}", "{'save': 'approve', 'wipe': 'run'}"),
                                   ('zoom', "AGENT_POLICY = 'run'", "AGENT_POLICY = 'approve'")):
                source = (commands / f'{name}.py').read_text(encoding='utf-8')
                self.assertEqual(source.count(old), 1, (name, old))
                (commands / f'{name}.py').write_text(source.replace(old, new), encoding='utf-8')
            with mock.patch.object(Command_Metadata, 'command_directories', return_value=[commands]):
                changed = unreviewed()
        self.assertEqual(set(changed) ^ set(RUNS_WITHOUT_REVIEW), {'zrunner', 'save', 'zoom'})
        self.assertEqual(changed['select'], {'*', 'wipe'})
        self.assertNotEqual(changed, RUNS_WITHOUT_REVIEW)


def read(source):
    return read_declarations(ast.parse(source))


class ReadingDeclarationsTests(unittest.TestCase):
    """A module's declarations are read from its source, by literals only; what cannot be read is not obeyed."""

    FULL = '''
from desktop.Command_Metadata import argument, choice, entry, EXPRESSION, plain_choices

AGENT_POLICY = {'default': 'run', 'first_argument': {'cancel': 'approve', '': 'run'}}
CATALOG_ENTRY = entry('Does a thing.',
    argument('action', 'What. ' + EXPRESSION, choice('list'), choice('remove', 'delete'), *plain_choices('a', 'b')),
    requires=('alignment',), writes_files=True)
AGENT_PROMPT_ORDER = 40
AGENT_PROMPT = """
`thing [ACTION]`
- Does a thing.
"""


def run(viewer, args):
    return None
'''

    def test_every_declaration_is_read_with_its_helpers_and_constants(self):
        declared = read(self.FULL)
        self.assertTrue(declared.has_run)
        self.assertEqual(declared.agent_policy, {'default': 'run', 'first_argument': {'cancel': 'approve', '': 'run'}})
        self.assertEqual(declared.catalog_entry['summary'], 'Does a thing.')
        self.assertEqual(declared.catalog_entry['requires'], ['alignment'])
        self.assertEqual((declared.catalog_entry['writes_files'], declared.catalog_entry['opens_interface']), (True, False))
        action = declared.catalog_entry['arguments'][0]
        self.assertTrue(action['description'].startswith('What. Boolean node selection'))
        self.assertEqual([item['value'] for item in action['choices']], ['list', 'remove', 'a', 'b'])
        self.assertEqual(action['choices'][1]['aliases'], ['delete'])
        self.assertEqual(declared.agent_prompt_order, 40)
        self.assertEqual(declared.agent_prompt, "\n`thing [ACTION]`\n- Does a thing.\n")

    def test_what_python_evaluates_and_what_is_read_are_one_value(self):
        # The module imports the helpers it is written with and runs: the value it gets is the value read.
        namespace = {}
        exec(compile(self.FULL, 'full', 'exec'), namespace)
        declared = read(self.FULL)
        self.assertEqual(namespace['CATALOG_ENTRY'], declared.catalog_entry)
        self.assertEqual(namespace['AGENT_PROMPT'], declared.agent_prompt)
        # ... and so does each shipped command: its import of the helpers and its declarations, run alone.
        for name in SHIPPED:
            with self.subTest(command=name):
                tree = ast.parse((COMMANDS_DIR / f'{name}.py').read_text(encoding='utf-8'))
                keep = [statement for statement in tree.body
                        if (isinstance(statement, ast.ImportFrom) and statement.module == 'desktop.Command_Metadata')
                        or (isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name)
                            and statement.targets[0].id in DECLARATIONS)]
                values = {}
                exec(compile(ast.Module(keep, []), name, 'exec'), values)
                found = read_declarations(tree)
                self.assertEqual(values['CATALOG_ENTRY'], found.catalog_entry)
                run_as = values['AGENT_POLICY']
                self.assertEqual({'default': run_as, 'first_argument': {}} if isinstance(run_as, str) else run_as,
                                 found.agent_policy)
                self.assertEqual(values.get('AGENT_PROMPT'), found.agent_prompt)
                self.assertEqual(values.get('AGENT_PROMPT_ORDER'), found.agent_prompt_order)

    def test_a_plain_dictionary_is_an_entry_too_and_a_word_is_a_policy(self):
        declared = read("CATALOG_ENTRY = {'summary': 'Plain.', 'arguments': [{'name': 'n', 'description': 'd', "
                        "'choices': [{'value': 'v', 'aliases': []}]}]}\nAGENT_POLICY = 'refuse'\ndef run(v, a): pass\n")
        self.assertEqual(declared.catalog_entry, entry('Plain.', {'name': 'n', 'description': 'd',
                                                                   'choices': [{'value': 'v', 'aliases': []}]}))
        self.assertEqual(declared.agent_policy, {'default': 'refuse', 'first_argument': {}})

    def test_nothing_declared_is_nothing(self):
        declared = read('def run(viewer, args):\n    return None\n')
        self.assertEqual(declared[1:], (None, None, None, None))
        self.assertTrue(declared.has_run)
        self.assertFalse(read('VALUE = 1\n').has_run)

    def test_a_policy_that_cannot_be_read_is_never_run(self):
        # Each is a way to ask for 'run' that must not get it: the command is left to wait for approval.
        unreadable = {
            'a word that is no kind': "AGENT_POLICY = 'sure'",
            'a kind in capitals': "AGENT_POLICY = 'Run'",
            'a default that is no kind': "AGENT_POLICY = {'default': 'ok'}",
            'a missing default': "AGENT_POLICY = {'first_argument': {'x': 'run'}}",
            'an unknown key': "AGENT_POLICY = {'default': 'run', 'then': 'refuse'}",
            'a word that is no kind in a form': "AGENT_POLICY = {'default': 'run', 'first_argument': {'x': 'sure'}}",
            'a form with a capital, which never matches': "AGENT_POLICY = {'default': 'run', 'first_argument': {'Cancel': 'approve'}}",
            'a form with a space': "AGENT_POLICY = {'default': 'run', 'first_argument': {'two words': 'approve'}}",
            'a form that is not a string': "AGENT_POLICY = {'default': 'run', 'first_argument': {1: 'approve'}}",
            'first_argument that is not a dictionary': "AGENT_POLICY = {'default': 'run', 'first_argument': ['x']}",
            'a call': "AGENT_POLICY = policy('run')",
            'a name': "RUN = 'run'\nAGENT_POLICY = RUN",
            'a conditional': "AGENT_POLICY = 'run' if True else 'approve'",
            'a computed word': "AGENT_POLICY = 'ru' + 'n'.upper()",
            'a function call of the module': "def verdict():\n    return 'run'\nAGENT_POLICY = verdict()",
            'an f-string': "AGENT_POLICY = f'{1}'",
            'a list': "AGENT_POLICY = ['run']",
            'None': "AGENT_POLICY = None",
            'a bare annotation': "AGENT_POLICY: str",
            'an assignment that follows with something unreadable': "AGENT_POLICY = 'run'\nAGENT_POLICY = compute()",
            'a second assignment inside an if': "AGENT_POLICY = 'run'\nif True:\n    AGENT_POLICY = 'approve'",
            'only an assignment inside an if': "if True:\n    AGENT_POLICY = 'run'",
            'only an assignment inside a try': "try:\n    AGENT_POLICY = 'run'\nexcept Exception:\n    pass",
            'one of several targets': "OTHER = AGENT_POLICY = 'run'",
            'a tuple target': "OTHER, AGENT_POLICY = 1, 'run'",
            'an augmented assignment': "AGENT_POLICY = 'run'\nAGENT_POLICY += 'x'",
            'a change in a function': "AGENT_POLICY = 'run'\ndef change():\n    global AGENT_POLICY\n    AGENT_POLICY = 'refuse'",
            'a function parameter of that name': "AGENT_POLICY = 'run'\ndef f(AGENT_POLICY):\n    return AGENT_POLICY",
            'a loop variable': "AGENT_POLICY = 'run'\nfor AGENT_POLICY in ('refuse',):\n    pass",
            'a deletion': "AGENT_POLICY = 'run'\ndel AGENT_POLICY",
            'an import of that name': "AGENT_POLICY = 'run'\nfrom elsewhere import AGENT_POLICY",
            'an import as that name': "AGENT_POLICY = 'run'\nimport elsewhere as AGENT_POLICY",
        }
        for why, source in unreadable.items():
            with self.subTest(why):
                self.assertIsNone(read(source + '\ndef run(viewer, args):\n    return None\n').agent_policy)

    def test_the_last_plain_assignment_is_the_one_python_ends_with(self):
        self.assertEqual(read("AGENT_POLICY = 'approve'\nAGENT_POLICY = 'run'")[2]['default'], 'run')
        self.assertEqual(read("AGENT_POLICY = 'run'\nAGENT_POLICY = 'refuse'")[2]['default'], 'refuse')
        self.assertEqual(read("AGENT_POLICY: str = 'run'")[2]['default'], 'run')

    def test_nothing_but_literals_constants_and_the_catalogs_helpers_is_evaluated(self):
        unreadable = [
            "CATALOG_ENTRY = __import__('os').getcwd()",
            "CATALOG_ENTRY = entry(open('x').read())",
            "CATALOG_ENTRY = entry('a' * 3)",
            "CATALOG_ENTRY = entry('a', *argument_list)",
            "CATALOG_ENTRY = entry(**{'summary': 'x'})",
            "CATALOG_ENTRY = entry(f'{1}')",
            "CATALOG_ENTRY = entry('x', argument('n', 'd', *[choice(s) for s in 'ab']))",
            "CATALOG_ENTRY = entry(UNKNOWN)",
            "CATALOG_ENTRY = entry(module.attribute)",
            "CATALOG_ENTRY = entry('x').copy()",
            "CATALOG_ENTRY = {**entry('x')}",
            "CATALOG_ENTRY = entry('x', requires=SOME)",
            "CATALOG_ENTRY = entry(b'bytes')",
            "CATALOG_ENTRY = entry(1j)",
            "CATALOG_ENTRY = entry('a', 'b')",
            "CATALOG_ENTRY = entry('x', writes_files='yes')",
            "CATALOG_ENTRY = entry('x', unknown=True)",
            "CATALOG_ENTRY = entry('   ')",
            "CATALOG_ENTRY = entry('x', argument('', 'd'))",
            "CATALOG_ENTRY = entry('x', argument('n', 'd', choice(1)))",
            "CATALOG_ENTRY = entry('x', argument('n', 'd', choice('v', 1)))",
            "CATALOG_ENTRY = entry('x', requires=(1,))",
            "CATALOG_ENTRY = 'a summary'",
            "CATALOG_ENTRY = {'summary': 'x', 'extra': 1}",
            "CATALOG_ENTRY = {'summary': 'x', 'arguments': [{'name': 'n'}]}",
            "CATALOG_ENTRY = 'x' + 1",
        ]
        for source in unreadable:
            with self.subTest(source[:60]):
                self.assertIsNone(read(source + '\ndef run(viewer, args):\n    return None\n').catalog_entry)

    def test_an_expression_nested_beyond_what_can_be_evaluated_is_unreadable_not_an_error(self):
        deep = ast.Constant('x')
        for _ in range(5000):
            deep = ast.BinOp(deep, ast.Add(), ast.Constant('y'))
        tree = ast.Module([ast.Assign([ast.Name('CATALOG_ENTRY', ast.Store())], deep, lineno=1)], [])
        self.assertIsNone(read_declarations(tree).catalog_entry)

    def test_a_prompt_and_its_order_must_be_plain_text_and_a_finite_number(self):
        for source, text, order in (("AGENT_PROMPT = ''", None, None), ("AGENT_PROMPT = '   '", None, None),
                                    ("AGENT_PROMPT = 5", None, None), ("AGENT_PROMPT = b'x'", None, None),
                                    ("AGENT_PROMPT = 'a' + 'b'", 'ab', None), ("AGENT_PROMPT = 'a' 'b'", 'ab', None),
                                    ("AGENT_PROMPT_ORDER = True", None, None), ("AGENT_PROMPT_ORDER = '3'", None, None),
                                    ("AGENT_PROMPT_ORDER = 1e999", None, None), ("AGENT_PROMPT_ORDER = -5", None, -5),
                                    ("AGENT_PROMPT_ORDER = 2.5", None, 2.5), ("AGENT_PROMPT_ORDER = 3 * 2", None, None)):
            with self.subTest(source):
                declared = read(source)
                self.assertEqual((declared.agent_prompt, declared.agent_prompt_order), (text, order))

    def test_the_declarations_of_a_file_follow_the_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'zchanging.py'
            path.write_text("AGENT_POLICY = 'run'\ndef run(v, a): pass\n", encoding='utf-8')
            self.assertEqual(declarations_of(path).agent_policy['default'], 'run')
            path.write_text("AGENT_POLICY = 'refuse'\ndef run(v, a): pass\n", encoding='utf-8')
            self.assertEqual(declarations_of(path).agent_policy['default'], 'refuse')
            path.write_text("AGENT_POLICY = (\n", encoding='utf-8')
            self.assertEqual(declarations_of(path)[1:], (None, None, None, None))
            path.unlink()
            self.assertFalse(declarations_of(path).has_run)

    def test_reading_runs_nothing_of_the_module(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'ran'
            source = f"open({str(marker)!r}, 'w').close()\nAGENT_POLICY = 'run'\ndef run(v, a): pass\n"
            path = Path(folder) / 'zeffect.py'
            path.write_text(source, encoding='utf-8')
            self.assertEqual(declarations_of(path).agent_policy['default'], 'run')
            self.assertFalse(marker.exists())


# A plugin as it is dropped into src/commands: one that declares nothing, one that declares it all, and one whose
# declarations cannot be read.
PLAIN = '''"""Scratch command that declares nothing."""
def print_help():
    print("Usage: zplain <thing>\\n  or: zplain reset")


def run(viewer, args):
    return None
'''
FULL = '''"""Scratch command that declares everything."""
from desktop.Command_Metadata import argument, choice, entry

# --- What this command declares about itself ---
AGENT_POLICY = {'default': 'run', 'first_argument': {'wipe': 'approve', 'nuke': 'refuse', '': 'run'}}
CATALOG_ENTRY = entry('Declared scratch command.',
    argument('mode', 'How it goes.', choice('fast', 'quick'), choice('slow')),
    requires=('alignment',), writes_files=True)
AGENT_PROMPT_ORDER = 25
AGENT_PROMPT = """
`zfull [fast|quick|slow]`
- Does the declared thing.
"""


def print_help():
    print("Usage: zfull [mode]")


def run(viewer, args):
    return None
'''
BROKEN = '''"""Scratch command whose declarations cannot be used."""
AGENT_POLICY = {'default': 'run', 'first_argument': {'Cancel': 'approve'}}
CATALOG_ENTRY = entry(summary_of_the_command())
AGENT_PROMPT = compute_the_prompt()


def run(viewer, args):
    return None
'''


class PluginDeclarationsTests(unittest.TestCase):
    """A module dropped into src/commands is catalogued and handled by what it declares: no edit to the catalog code,
    the Agent's instructions or any table of the tests is needed."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.commands = Path(cls.folder.name) / 'commands'
        shutil.copytree(COMMANDS_DIR, cls.commands, ignore=shutil.ignore_patterns('__pycache__'))
        for name, source in {'zplain': PLAIN, 'zfull': FULL, 'zbroken': BROKEN}.items():
            (cls.commands / f'{name}.py').write_text(source, encoding='utf-8')
        cls.entries = {e['command']: e for e in command_catalog(commands_dir=cls.commands)['commands']}

    def setUp(self):
        self.enterContext(mock.patch.object(Command_Metadata, 'command_directories', return_value=[self.commands]))

    def test_a_module_that_declares_nothing_gets_the_defaults(self):
        found = self.entries['zplain']
        self.assertIs(found['metadata_declared'], False)
        self.assertEqual(found['summary'], 'Scratch command that declares nothing.')
        self.assertEqual((found['arguments'], found['requires']), ([], []))
        self.assertTrue(found['writes_files'] and found['opens_interface'])
        self.assertEqual(found['agent_policy'], {'default': 'approve', 'first_argument': {}})
        self.assertEqual(found['syntax'], ['zplain <thing>', 'zplain reset'])
        for text in ('zplain', 'zplain reset', 'ZPLAIN anything'):
            self.assertEqual(agent_policy(text), 'approve')
        self.assertNotIn('zplain', AGENT_POLICY)
        self.assertNotIn('zplain', agent_policy_summary()[0]['run'] + agent_policy_summary()[0]['approve'])
        self.assertEqual([(item.name, item.text, item.order) for item in agent_prompt_items() if item.name == 'zplain'],
                         [('zplain', None, None)])

    def test_a_module_that_declares_everything_shows_it_all(self):
        found = self.entries['zfull']
        self.assertNotIn('metadata_declared', found)
        self.assertEqual(found['summary'], 'Declared scratch command.')
        self.assertEqual(found['requires'], ['alignment'])
        self.assertEqual((found['writes_files'], found['opens_interface']), (True, False))
        self.assertEqual([(a['name'], [c['value'] for c in a['choices']]) for a in found['arguments']],
                         [('mode', ['fast', 'slow']), ('help', ['help'])])
        self.assertEqual(found['arguments'][0]['choices'][0], {'value': 'fast', 'aliases': ['quick']})
        self.assertEqual(found['agent_policy'], {'default': 'run', 'first_argument': {'wipe': 'approve', 'nuke': 'refuse', '': 'run'}})
        self.assertEqual(found['syntax'], ['zfull [mode]'])
        for text, expected in (('zfull', 'run'), ('zfull fast', 'run'), ('zfull wipe now', 'approve'), ('ZFULL Nuke', 'refuse')):
            self.assertEqual(agent_policy(text), expected)
        self.assertEqual(AGENT_POLICY['zfull'], found['agent_policy'])
        plain, by_form = agent_policy_summary()
        self.assertEqual([entry for entry in by_form if entry[0] == 'zfull'],
                         [('zfull', 'run', {'wipe': 'approve', 'nuke': 'refuse'})])
        items = {item.name: item for item in agent_prompt_items()}
        self.assertEqual((items['zfull'].order, items['zfull'].text.strip().splitlines()[0]), (25, '`zfull [fast|quick|slow]`'))
        # Its entry is the one the main program's catalog serves (what the VR viewer's help reads) where its folder is.
        with mock.patch.object(Command_Metadata, 'COMMANDS_DIR', self.commands):
            self.assertEqual({key: value for key, value in COMMAND_METADATA['zfull'].items()},
                             {key: found[key] for key in ('summary', 'arguments', 'requires', 'writes_files',
                                                          'opens_interface', 'agent_policy')})
            self.assertEqual(get_command_metadata('zfull'), COMMAND_METADATA['zfull'])
        with self.assertRaises(KeyError):
            get_command_metadata('zplain')

    def test_a_declaration_that_cannot_be_used_counts_as_none(self):
        found = self.entries['zbroken']
        self.assertIs(found['metadata_declared'], False)
        self.assertEqual(found['agent_policy'], {'default': 'approve', 'first_argument': {}})
        self.assertEqual(agent_policy('zbroken'), 'approve')
        self.assertEqual(agent_policy('zbroken cancel'), 'approve')
        self.assertNotIn('zbroken', AGENT_POLICY)
        self.assertEqual([(i.name, i.text, i.order) for i in agent_prompt_items() if i.name == 'zbroken'],
                         [('zbroken', None, None)])

    def test_the_shipped_commands_are_untouched_by_the_plugins(self):
        shipped = {e['command']: e for e in command_catalog()['commands']}
        for name in SHIPPED:
            self.assertEqual(self.entries[name], shipped[name])
            self.assertEqual(agent_policy(name), TODAYS_POLICY[name][1].get('', TODAYS_POLICY[name][0]))

    def test_the_catalog_is_json_as_the_mcp_server_serves_it(self):
        json.dumps(command_catalog(commands_dir=self.commands))
        detailed = command_catalog('zfull', commands_dir=self.commands)['commands'][0]
        self.assertIn('Usage: zfull [mode]', detailed['help'])

    def test_defaults_are_never_the_shared_constant(self):
        self.entries['zplain']['agent_policy']['default'] = 'run'
        self.assertEqual(default_command_metadata('x')['agent_policy'], {'default': 'approve', 'first_argument': {}})
        self.assertEqual(default_command_metadata('x', {'default': 'refuse', 'first_argument': {}})['agent_policy']['default'],
                         'refuse')
        self.assertEqual(UNLISTED_POLICY['default'], 'approve')
        self.assertEqual(agent_prompt_items.__module__, 'desktop.Command_Metadata')

if __name__ == '__main__':
    unittest.main()
