"""Static Viewer command catalog (desktop.Command_Metadata, desktop.Viewer_Inspection.command_catalog).

Covers the catalog's argument metadata and finite choices, which must be literal
tokens of each command's parser, and the syntax and help text it extracts from
the command sources without importing or executing any handler.
"""
import ast
from pathlib import Path
import sys
import unittest
from unittest import mock

COMMANDS_DIR = Path(__file__).resolve().parents[1] / 'src' / 'commands'
sys.path.insert(0, str(COMMANDS_DIR.parent))
from desktop.Command_Metadata import COMMAND_METADATA, get_command_metadata  # noqa: E402
from desktop.Viewer_Inspection import command_catalog, _command_syntax  # noqa: E402


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
        self.assertEqual(len(general['commands']), 24)
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


if __name__ == '__main__':
    unittest.main()
