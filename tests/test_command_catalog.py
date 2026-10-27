"""Static Viewer command catalog (desktop.Command_Metadata, desktop.Viewer_Inspection.command_catalog).

Covers the catalog's argument metadata and finite choices, which must be literal
tokens of each command's parser, the syntax and help text it extracts from the
command sources without importing or executing any handler, and the policy for
commands in Agent replies.
"""
import ast
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

COMMANDS_DIR = Path(__file__).resolve().parents[1] / 'src' / 'commands'
sys.path.insert(0, str(COMMANDS_DIR.parent))
from desktop.Command_Metadata import (  # noqa: E402
    AGENT_POLICY, COMMAND_METADATA, UNLISTED_POLICY, agent_policy, default_command_metadata, entry, get_command_metadata)
from desktop.Viewer_Inspection import command_catalog, _command_syntax  # noqa: E402


def string_literals(command):
    source = (COMMANDS_DIR / (command + '.py')).read_text(encoding='utf-8')
    return {n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


class MetadataTests(unittest.TestCase):
    def test_complete_independent_metadata_and_finite_choice_sources(self):
        catalog = command_catalog()['commands']
        self.assertEqual(set(COMMAND_METADATA), {c['command'] for c in catalog})
        for entry in catalog:
            name = entry['command']
            with self.subTest(command=name):
                self.assertTrue(entry['summary'])
                self.assertIsNone(entry['help'])
                self.assertEqual(len({a['name'] for a in entry['arguments']}), len(entry['arguments']))
                source = (COMMANDS_DIR / (name + '.py')).read_text(encoding='utf-8')
                literals = {n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
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
        self.assertEqual(len(general['commands']), 28)
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
            {name for name, metadata in COMMAND_METADATA.items() if 'alignment' in metadata['requires']},
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
        self.assertEqual(len(self.shipped), 28)
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

    def test_a_plugin_can_declare_its_own_entry_and_flags_without_touching_the_catalog_code(self):
        declared = {'zbare': entry('Scratch command that only reads.', writes_files=False, opens_interface=False)}
        declared['zbare']['agent_policy'] = {'default': 'run', 'first_argument': {}}
        with mock.patch.dict(COMMAND_METADATA, declared):
            found = {e['command']: e for e in command_catalog(commands_dir=self.commands)['commands']}['zbare']
        self.assertEqual((found['summary'], found['writes_files'], found['opens_interface']),
                         ('Scratch command that only reads.', False, False))
        self.assertEqual(found['agent_policy']['default'], 'run')
        self.assertNotIn('metadata_declared', found)

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
        self.assertEqual({n for n, m in COMMAND_METADATA.items() if m['writes_files']},
                         {'esmfold', 'export', 'label', 'logo', 'meta', 'print', 'run', 'save', 'select'})
        self.assertEqual({n for n, m in COMMAND_METADATA.items() if m['opens_interface']},
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


class AgentPolicyTests(unittest.TestCase):
    """How a command in an Agent reply is handled; Viewer_Command_Portal applies it."""

    def test_every_command_has_a_policy_in_the_catalog(self):
        self.assertEqual(set(AGENT_POLICY), set(COMMAND_METADATA))
        for entry in command_catalog()['commands']:
            with self.subTest(command=entry['command']):
                rule = entry['agent_policy']
                self.assertEqual(rule, AGENT_POLICY[entry['command']])
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


if __name__ == '__main__':
    unittest.main()
