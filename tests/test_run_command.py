"""The run command (commands/run.py): the commands a .txt command file holds, or
a Python command script prints, reach the Viewer unchanged. Typed into the
Viewer, run.py runs a script itself; from a command-portal request (an MCP
client or the agent page) it hands a script to
Viewer_Worker_Tracking.ScriptTracker, and a .txt file's lines to
context.children. Typed, a script runs every line and reports how many failed."""

import codecs
import io
import os
import subprocess
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.viewer_fixtures import PortalFixture
from PySide6 import QtWidgets
import Command_Engine as ce


# 'é' is in the Windows ANSI code page (cp1252) and used to arrive dropped or as
# U+FFFD; 'α' is not, and the script died with UnicodeEncodeError ('charmap').
# The last line checks that the script still sees the Viewer's environment.
NON_ASCII_SCRIPT = '''import os
print('select "café"')
print('select "α-amylase"')
print('select "' + os.environ['RUN_SCRIPT_MARKER'] + '"')
'''

# A byte that is not UTF-8 (cp1252 'é'), written past the script's text layer,
# as a process the script starts could write it.
RAW_BYTE_SCRIPT = r'''import sys
sys.stdout.buffer.write(b'select "caf\xe9"\n')
'''

# A command file as Windows editors save it. 'α' is not in cp1252, so only the
# Unicode encodings can hold all of it.
COMMANDS = 'select "café" // déjà vu\r\nselect "α-amylase"\r\n'
EXPECTED = ['select "café"', 'select "α-amylase"']
# Printed when a file falls back to the ANSI code page.
FALLBACK_NOTE = 'is not UTF-8'

# Comment lines, and '//' inside a word, which is no comment: a URL, or a quoted
# pattern. `bogus` is no command, so these lines are dispatched and fail harmlessly.
COMMENTED = (
    '# a comment\n'
    'bogus C://data/x.csv\n'
    'bogus "a//b" // a trailing comment\n'
    '    // an indented comment\n'
    '   # an indented hash line\n'
    'bogus a //b\n'
)
COMMENTED_EXPECTED = ['bogus C://data/x.csv', 'bogus "a//b"', 'bogus a']


@contextmanager
def parent_environment(overrides):
    """Patch os.environ with ``overrides``, minus any Python encoding variable they don't set."""
    with mock.patch.dict(os.environ, overrides):
        for name in ('PYTHONIOENCODING', 'PYTHONUTF8'):
            if name not in overrides:
                os.environ.pop(name, None)
        yield


class RunPaths(PortalFixture):
    """Runs `run` both ways, with the (mocked) file dialog choosing self.script."""

    def setUp(self):
        super().setUp()
        self.viewer.canvas = SimpleNamespace(native=None)  # the file dialog's parent
        self.script = Path(self.directory.name) / 'commands.py'

    def choose_script(self):
        return mock.patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(str(self.script), ''))

    def typed(self):
        """Run `run` as typed into the Viewer; return the commands run.py dispatched and what it printed."""
        printed = io.StringIO()
        with self.choose_script(), redirect_stdout(printed), \
                mock.patch.object(ce, '_dispatch_user_command', wraps=ce._dispatch_user_command) as dispatch:
            self.viewer.process_command('run', record_history=False)
        commands = [call.args[1] for call in dispatch.call_args_list]
        self.assertEqual(commands[0], 'run')
        return commands[1:], printed.getvalue()

    def requested(self):
        """Submit `run` through the command portal; return its child commands, and the request and its stderr."""
        # The file dialog only opens for a visible Viewer.
        with self.choose_script(), redirect_stdout(io.StringIO()), \
                mock.patch.dict(os.environ, {'SSN_VIEWER_HEADLESS': '0', 'QT_QPA_PLATFORM': ''}):
            request_id = self.portal.submit(str(len(self.portal.requests)), 'run')['request_id']
            result = self.finish(request_id)
        children = [record['command'] for record in result['commands'] if record['parent_command_id']]
        return children, (result, self.portal.read_output(request_id, 'stderr')['text'])


class ScriptOutputTests(RunPaths, unittest.TestCase):
    """A script's stdout is decoded as UTF-8, so the script must print UTF-8 whatever the Viewer's environment says."""

    def test_non_ascii_commands_arrive_whatever_the_parent_encoding(self):
        self.script.write_text(NON_ASCII_SCRIPT, encoding='utf-8')
        expected = ['select "café"', 'select "α-amylase"', 'select "inherited"']
        for label, overrides in (
                # A piped child's Windows default (the ANSI code page), forced on any platform.
                ('PYTHONIOENCODING=cp1252', {'PYTHONIOENCODING': 'cp1252'}),
                # This platform's own default, which Claude Code's shells mask with PYTHONIOENCODING.
                ('no encoding variables', {})):
            with parent_environment({'RUN_SCRIPT_MARKER': 'inherited', **overrides}):
                for path in (self.typed, self.requested):
                    with self.subTest(environment=label, path=path.__name__):
                        commands, diagnostics = path()
                        self.assertEqual(commands, expected, diagnostics)

    def test_bytes_that_are_not_utf8_stay_visible(self):
        # Dropping the byte would run select "caf", which also matches caffeine.
        self.script.write_text(RAW_BYTE_SCRIPT, encoding='utf-8')
        for path in (self.typed, self.requested):
            with self.subTest(path=path.__name__):
                commands, diagnostics = path()
                self.assertEqual(commands, ['select "caf�"'], diagnostics)


class ScriptStdinTests(RunPaths, unittest.TestCase):
    """A script has no stdin: input() fails at once instead of waiting on the Viewer's own."""

    def test_a_script_calling_input_fails_instead_of_waiting(self):
        self.script.write_text('answer = input()\nprint(\'select "\' + answer + \'"\')\n', encoding='utf-8')
        for path in (self.typed, self.requested):
            with self.subTest(path=path.__name__), mock.patch('subprocess.run', wraps=subprocess.run) as run:
                commands, diagnostics = path()
            self.assertEqual(commands, [], diagnostics)
            self.assertIs(run.call_args.kwargs['stdin'], subprocess.DEVNULL)


class ScriptCommentTests(RunPaths, unittest.TestCase):
    """A comment is a # line or a // after whitespace; a // inside a word stays in the command."""

    def test_comments_are_dropped_and_a_double_slash_inside_a_word_is_kept(self):
        for name, data in (('commands.txt', COMMENTED),
                           ('commands.py', f'import sys\nsys.stdout.write({COMMENTED!r})\n')):
            self.script = Path(self.directory.name) / name
            self.script.write_text(data, encoding='utf-8')
            for path in (self.typed, self.requested):
                with self.subTest(script=name, path=path.__name__):
                    commands, diagnostics = path()
                    self.assertEqual(commands, COMMENTED_EXPECTED, diagnostics)

    def test_script_command(self):
        for line, expected in (
                ('select "a"', 'select "a"'),
                ('  select "a"  \r\n', 'select "a"'),
                ('select "a" // note', 'select "a"'),
                ('select "a"\t// note', 'select "a"'),
                ('select "a" //note', 'select "a"'),
                ('meta C://data/x.csv', 'meta C://data/x.csv'),
                ('select "a//b"', 'select "a//b"'),
                ('// whole line', None), ('  // whole line', None),
                ('# whole line', None), ('   #cluster_1# red', None),
                ('', None), ('   \n', None)):
            with self.subTest(line=line):
                self.assertEqual(ce.script_command(line), expected)


class TypedRunOutcomeTests(RunPaths, unittest.TestCase):
    """Typed, run runs every line of a script, and ends in a failure when any line failed."""

    def setUp(self):
        super().setUp()
        self.script = Path(self.directory.name) / 'commands.txt'

    def run_script(self, *lines):
        """Run the lines as a typed script; return the commands dispatched, what was printed, and run's outcome mocks."""
        self.script.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        with mock.patch.object(ce, 'command_failed', wraps=ce.command_failed) as failed, \
                mock.patch.object(ce, 'command_succeeded', wraps=ce.command_succeeded) as succeeded:
            commands, printed = self.typed()
        return commands, printed, failed, succeeded

    def test_a_failed_line_does_not_stop_the_script_and_is_counted(self):
        for lines, expected in (
                # No command is named bogus; reset rejects its target.
                (('reset hide', 'bogus', 'reset colors'), '1 of 3 commands failed.'),
                (('reset bogus', 'reset hide', 'bogus'), '2 of 3 commands failed.'),
                (('bogus',), '1 of 1 command failed.')):
            with self.subTest(lines=lines):
                commands, printed, failed, succeeded = self.run_script(*lines)

                self.assertEqual(commands, list(lines))
                message = f'Batch execution finished: {expected}'
                self.assertEqual(str(failed.call_args.args[1]), message)
                self.assertIn(message, printed)
                self.assertNotIn('Batch execution completed', printed)
                self.assertFalse(any('Batch execution' in str(call.args[1]) for call in succeeded.call_args_list))

    def test_a_script_without_failures_keeps_the_success_message(self):
        commands, printed, failed, succeeded = self.run_script('reset hide', 'reset colors')

        self.assertEqual(commands, ['reset hide', 'reset colors'])
        failed.assert_not_called()
        self.assertEqual(str(succeeded.call_args.args[1]), 'Batch execution completed: 2 commands run.')
        self.assertIn('Batch execution completed: 2 commands run.', printed)

    def test_recorded_outcome_keeps_a_failure(self):
        with ce.recorded_outcome() as outcome:
            self.assertIsNone(outcome['status'])
            ce.command_succeeded(self.viewer)
            self.assertEqual(outcome['status'], 'succeeded')
            ce.command_failed(self.viewer, 'failed')
            ce.command_succeeded(self.viewer)
        self.assertEqual(outcome['status'], 'failed')
        # Outside the block nothing is recorded.
        ce.command_succeeded(self.viewer)
        self.assertEqual(outcome['status'], 'failed')


class CommandFileEncodingTests(RunPaths, unittest.TestCase):
    """A .txt command file is decoded as Notepad decodes it: by its byte-order mark,
    else as UTF-8 when it is UTF-8, else in the system's ANSI code page."""

    def setUp(self):
        super().setUp()
        self.script = Path(self.directory.name) / 'commands.txt'

    def assert_commands(self, data, expected, code_page='cp1252', note=False):
        """A command file holding ``data`` yields ``expected`` both ways when the ANSI
        code page is ``code_page``; the typed path prints the fallback note only if ``note``."""
        self.script.write_bytes(data)
        with mock.patch('locale.getencoding', return_value=code_page):
            for path in (self.typed, self.requested):
                with self.subTest(path=path.__name__):
                    commands, diagnostics = path()
                    self.assertEqual(commands, expected, diagnostics)
                    if path == self.typed:
                        self.assertEqual(FALLBACK_NOTE in diagnostics, note, diagnostics)

    def test_a_byte_order_mark_names_the_encoding(self):
        for label, data in (
                # Notepad's "UTF-8 with BOM"; PowerShell 5.1's Set-Content -Encoding UTF8.
                ('UTF-8', codecs.BOM_UTF8 + COMMANDS.encode('utf-8')),
                # PowerShell 5.1's > and Out-File default.
                ('UTF-16 LE', codecs.BOM_UTF16_LE + COMMANDS.encode('utf-16-le')),
                ('UTF-16 BE', codecs.BOM_UTF16_BE + COMMANDS.encode('utf-16-be')),
                # This mark begins with UTF-16 LE's.
                ('UTF-32 LE', codecs.BOM_UTF32_LE + COMMANDS.encode('utf-32-le')),
                ('UTF-32 BE', codecs.BOM_UTF32_BE + COMMANDS.encode('utf-32-be'))):
            with self.subTest(encoding=label):
                self.assert_commands(data, EXPECTED)

    def test_bytes_a_marked_file_cannot_decode_stay_visible(self):
        # Dropping the byte would run select "caf", which also matches caffeine.
        self.assert_commands(codecs.BOM_UTF8 + b'select "caf\xe9"\r\n', ['select "caf�"'])

    def test_utf8_without_a_mark_is_read_as_utf8(self):
        # Notepad's default today. Read as cp1252, 'é' would become 'Ã©'.
        self.assert_commands(COMMANDS.encode('utf-8'), EXPECTED)

    def test_other_files_are_read_in_the_ansi_code_page(self):
        for label, data, code_page, expected in (
                # Older Notepad's and PowerShell 5.1 Set-Content's default on Western Windows.
                ('cp1252', 'select "café" // déjà vu\r\n'.encode('cp1252'), 'cp1252', ['select "café"']),
                # The system's own code page, which isn't cp1252 everywhere.
                ('cp1251', 'select "фосфатаза"\r\n'.encode('cp1251'), 'cp1251', ['select "фосфатаза"']),
                # Linux and macOS: the locale encoding is UTF-8, so the byte shows as U+FFFD.
                ('UTF-8 locale', 'select "café"\r\n'.encode('cp1252'), 'utf-8', ['select "caf�"'])):
            with self.subTest(code_page=label):
                self.assert_commands(data, expected, code_page=code_page, note=True)

    def test_lines_end_at_any_newline(self):
        # As readlines() on a text-mode file splits them.
        self.assert_commands(b'select "a"\rselect "b"\nselect "c"\r\n',
                             ['select "a"', 'select "b"', 'select "c"'])


if __name__ == '__main__':
    unittest.main()
