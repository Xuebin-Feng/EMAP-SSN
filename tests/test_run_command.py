"""The run command (commands/run.py): the commands a Python command script
prints reach the Viewer unchanged. Typed into the Viewer, run.py runs the script
itself; from a command-portal request (an MCP client or the agent page) it hands
the script to Viewer_Worker_Tracking.ScriptTracker."""

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


@contextmanager
def parent_environment(overrides):
    """Patch os.environ with ``overrides``, minus any Python encoding variable they don't set."""
    with mock.patch.dict(os.environ, overrides):
        for name in ('PYTHONIOENCODING', 'PYTHONUTF8'):
            if name not in overrides:
                os.environ.pop(name, None)
        yield


class ScriptOutputTests(PortalFixture, unittest.TestCase):
    """A script's stdout is decoded as UTF-8, so the script must print UTF-8 whatever the Viewer's environment says."""

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


if __name__ == '__main__':
    unittest.main()
