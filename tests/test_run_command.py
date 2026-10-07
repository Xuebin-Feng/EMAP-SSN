"""The run command (commands/run.py): the commands a .txt command file holds, or
a Python command script prints, reach the Viewer unchanged. Typed into the
Viewer, run.py runs a script itself; from a command-portal request (an MCP
client or the agent page) it hands a script to
Viewer_Worker_Tracking.ScriptTracker, and a .txt file's lines to
context.children."""

import codecs
import io
import os
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
