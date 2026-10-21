"""Static Viewer command catalog (desktop.Command_Metadata, desktop.Viewer_Inspection.command_catalog).

Covers the catalog's argument metadata and finite choices, which must be literal
tokens of each command's parser, the syntax and help text it extracts from the
command sources without importing or executing any handler, and the policy for
commands in Agent replies.
"""
import ast
from pathlib import Path
import sys
import unittest
from unittest import mock

COMMANDS_DIR = Path(__file__).resolve().parents[1] / 'src' / 'commands'
sys.path.insert(0, str(COMMANDS_DIR.parent))
from desktop.Command_Metadata import AGENT_POLICY, COMMAND_METADATA, agent_policy, get_command_metadata  # noqa: E402
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
        self.assertEqual(len(general['commands']), 26)
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
        for name, metadata in COMMAND_METADATA.items():
            if 'alignment' in metadata['requires']:
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
