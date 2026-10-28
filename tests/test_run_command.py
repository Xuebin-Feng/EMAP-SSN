"""The run command (commands/run.py): the commands a .txt command file holds, or
a Python command script prints, reach the Viewer unchanged. Typed into the
Viewer, run.py runs a script itself; from a command-portal request (an MCP
client or the agent page) it hands a script to
Viewer_Worker_Tracking.ScriptTracker, and a .txt file's lines to
context.children. Both obey one rule: a script holds at most 1000 commands, and
the first command that fails ends the run. A Python script runs in its own
folder; typed, it runs in the background and its commands run when it ends.
A command may go on over several lines (a line ending in a backslash after
whitespace): Command_Engine.script_command_lines joins them for both kinds of
script, and a script whose last line continues is refused whole."""

import codecs
import gc
import importlib
import io
import os
import queue
import subprocess
import threading
import time
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import psutil
from tests.viewer_fixtures import PortalFixture, Viewer
from PySide6 import QtWidgets
import Command_Engine as ce
from utilities import Localization
from utilities.Localization import display_text


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

# A script that waits until go.txt exists beside it: a relative name, so it also
# shows the folder the script runs in. The deadline only ends a test that went wrong.
WAITING_SCRIPT = '''import os, time
deadline = time.monotonic() + 30
while not os.path.exists('go.txt') and time.monotonic() < deadline:
    time.sleep(.01)
print('select "one"')
'''

# A script that starts a process of its own, writes both process ids beside
# itself, and then waits: what `run stop` and closing the Viewer must end.
TREE_SCRIPT = """import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
with open('pids.part', 'w') as handle:
    handle.write(f'{os.getpid()} {child.pid}')
os.replace('pids.part', 'pids.txt')
time.sleep(120)
"""

# A command file as Windows editors save it. 'α' is not in cp1252, so only the
# Unicode encodings can hold all of it.
COMMANDS = 'select "café" // déjà vu\r\nselect "α-amylase"\r\n'
EXPECTED = ['select "café"', 'select "α-amylase"']
# Printed when a file falls back to the ANSI code page.
FALLBACK_NOTE = 'is not UTF-8'

# Comment lines, and '//' inside a word, which is no comment: a URL, or a quoted
# pattern. `bogus` is no command, so these lines fail if they run; the typed tests
# only dispatch them.
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

    def wait_for_script(self, timeout=30):
        """Process Qt events until the typed Python script has ended and its commands have run."""
        deadline = time.monotonic() + timeout
        while getattr(self.viewer, 'command_script_run', None) is not None:
            if time.monotonic() >= deadline:
                self.fail(f'The Python script was still running after {timeout} s')
            self.app.processEvents()
            time.sleep(.005)

    def typed(self, quick=False):
        """Run `run` as typed into the Viewer; return the commands run.py dispatched and what it printed.

        A Python script runs in the background, so this waits for it and for its commands to run.
        With quick, the commands after `run` are only recorded, as if each succeeded: a long script is fast."""
        printed = io.StringIO()
        real = ce._dispatch_user_command

        def dispatch(viewer, command, *args, **options):
            if quick and command != 'run':
                return ce.command_succeeded(viewer)
            return real(viewer, command, *args, **options)

        with self.choose_script(), redirect_stdout(printed), \
                mock.patch.object(ce, '_dispatch_user_command', side_effect=dispatch) as dispatched:
            self.viewer.process_command('run', record_history=False)
            self.wait_for_script()
        commands = [call.args[1] for call in dispatched.call_args_list]
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

    def test_a_python_script_that_prints_run_lines_has_them_dropped_typed_and_requested(self):
        # The run command says which lines a script may not hold, so the portal's script tracker names no command.
        self.script.write_text("print('run')\nprint('RUN again')\nprint('select \"one\"')\nprint('run stop')\n",
                               encoding='utf-8')
        for path in (self.typed, self.requested):
            with self.subTest(path=path.__name__):
                commands, diagnostics = path()
                self.assertEqual(commands, ['select "one"'], diagnostics)

    def test_the_tracker_keeps_every_line_unless_the_command_that_starts_it_says_otherwise(self):
        from Viewer_Command_Portal import ExecutionContext
        from Viewer_Worker_Tracking import ScriptTracker
        self.script.write_text("print('run')\nprint('select \"one\"')\n", encoding='utf-8')
        # A kept `run` line becomes a child command, which must not reach a dialog.
        no_dialog = mock.patch.object(QtWidgets.QFileDialog, 'getOpenFileName', side_effect=AssertionError('no dialog'))
        for keep, expected in ((lambda line: line != 'run', ['select "one"']), (None, ['run', 'select "one"'])):
            with self.subTest(keep=keep is not None), no_dialog, redirect_stdout(io.StringIO()):
                request_id = self.portal.submit(str(len(self.portal.requests)), 'run')['request_id']
                record = self.portal.requests[request_id]['commands'][0]
                record.update(dispatched=True, outcome='succeeded', status='running')
                ScriptTracker(ExecutionContext(self.portal, request_id, record), str(self.script), keep_command=keep)
                self.assertEqual(record['children'], [])
                deadline = time.monotonic() + 30
                while not record['children'] and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(.005)
                self.assertEqual([child['command'] for child in record['children']], expected)
                self.finish(request_id)

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
            with self.subTest(path=path.__name__), mock.patch('subprocess.Popen', wraps=subprocess.Popen) as run:
                commands, diagnostics = path()
            self.assertEqual(commands, [], diagnostics)
            self.assertIs(run.call_args.kwargs['stdin'], subprocess.DEVNULL)


class ScriptFolderTests(RunPaths, unittest.TestCase):
    """A Python script runs in its own folder, so a relative path in it names a file beside the script."""

    def test_a_script_runs_in_its_own_folder_typed_and_requested(self):
        folder = Path(self.directory.name) / 'scripts'
        folder.mkdir()
        self.script = folder / 'commands.py'
        # The folder the Viewer runs in is not the script's, so a script that
        # found its neighbours by luck would fail here.
        self.assertNotEqual(os.path.realpath(os.getcwd()), os.path.realpath(folder))
        self.script.write_text(
            "import os\n"
            "print('select \"' + os.path.realpath(os.getcwd()) + '\"')\n"
            "open('written_beside_the_script.txt', 'w').close()\n", encoding='utf-8')
        for path in (self.typed, self.requested):
            with self.subTest(path=path.__name__), mock.patch('subprocess.Popen', wraps=subprocess.Popen) as run:
                commands, diagnostics = path()
            self.assertEqual(commands, [f'select "{os.path.realpath(folder)}"'], diagnostics)
            self.assertEqual(os.path.realpath(run.call_args.kwargs['cwd']), os.path.realpath(folder))
            self.assertTrue((folder / 'written_beside_the_script.txt').exists())
            (folder / 'written_beside_the_script.txt').unlink()

    def test_a_relative_script_path_still_names_the_script(self):
        from Viewer_Worker_Tracking import start_command_script
        folder = Path(self.directory.name) / 'scripts'
        folder.mkdir()
        (folder / 'commands.py').write_text("import os\nprint(os.path.realpath(os.getcwd()))\n", encoding='utf-8')
        previous = os.getcwd()
        self.addCleanup(os.chdir, previous)  # Before the temporary folder is removed, so it is not the working folder then.
        os.chdir(self.directory.name)
        results = queue.Queue()

        start_command_script(os.path.join('scripts', 'commands.py'), SimpleNamespace(emit=results.put))

        result = results.get(timeout=30)
        self.assertEqual(result.returncode, 0, getattr(result, 'stderr', result))
        self.assertEqual(result.stdout.strip(), os.path.realpath(folder))


class ScriptCommentTests(RunPaths, unittest.TestCase):
    """A comment is a # line or a // after whitespace; a // inside a word stays in the command."""

    def test_comments_are_dropped_and_a_double_slash_inside_a_word_is_kept(self):
        for name, data in (('commands.txt', COMMENTED),
                           ('commands.py', f'import sys\nsys.stdout.write({COMMENTED!r})\n')):
            self.script = Path(self.directory.name) / name
            self.script.write_text(data, encoding='utf-8')
            for path, run in (('typed', lambda: self.typed(quick=True)), ('requested', self.requested)):
                with self.subTest(script=name, path=path):
                    commands, diagnostics = run()
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

    def test_script_command_reads_one_line_and_knows_nothing_of_continuation(self):
        # The VR terminal's run (opt_vr) aliases this function and calls it on each line, so a line that ends in
        # a backslash must come out as it always did: script_command_lines is the one that joins lines.
        for line, expected in (
                ('color #A# red \\', 'color #A# red \\'),
                ('color #A# red \\  \t\r\n', 'color #A# red \\'),
                ('\\', '\\'),
                ('#B# blue', None), ('  #B# blue \\', None),
                ('alignment C:\\folder\\', 'alignment C:\\folder\\'),
                ('color #A# red \\ // first part', 'color #A# red \\'),
                ('color #A# red // not \\', 'color #A# red')):
            with self.subTest(line=line):
                self.assertEqual(ce.script_command(line), expected)


# The command of a manuscript figure, over two lines; the second starts with a #label#.
FIGURE_LINES = ['color #BCAT# cyan #C-DAAT# pink #NC-DAAT# green \\', '#R-ATA# red #ADCL# magenta']
FIGURE_COMMAND = 'color #BCAT# cyan #C-DAAT# pink #NC-DAAT# green #R-ATA# red #ADCL# magenta'
DANGLING_MESSAGE = ('Error: Line {line} of the script ends with a backslash, but no line follows it for the '
                    'command to continue on. None were run.')


class ScriptContinuationTests(unittest.TestCase):
    """script_command_lines joins a command that goes on over several lines, and leaves every other backslash alone.

    A line goes on at the next when its last non-whitespace character is a backslash that follows whitespace
    (or is the whole line); the backslash and the line break become one space."""

    def commands(self, *lines):
        return ce.script_command_lines(list(lines))

    def test_the_figure_command_is_one_command(self):
        self.assertEqual(self.commands(*FIGURE_LINES), [FIGURE_COMMAND])
        # Whatever the line ends in, and whether or not it kept its line break.
        self.assertEqual(self.commands(*[line + '\n' for line in FIGURE_LINES]), [FIGURE_COMMAND])
        self.assertEqual(self.commands(*[line + '\r\n' for line in FIGURE_LINES]), [FIGURE_COMMAND])

    def test_a_backslash_after_no_whitespace_is_not_a_continuation(self):
        # A Windows path can never be misread as one, and nor can the text a command holds.
        for line in ('alignment C:\\data\\aln.fasta', 'alignment C:\\folder\\', 'alignment C:\\folder\\  \t',
                     'select "a\\b"', 'select "a\\"', 'label {k=a\\b}', 'label {k=a\\}', 'select #a\\b#',
                     'select @C:\\lists\\x.txt@', 'select @x\\', 'color #A# red \\\\', 'color red\\'):
            with self.subTest(line=line):
                self.assertEqual(self.commands(line, 'select "next"'), [line.strip(), 'select "next"'])
                self.assertEqual(self.commands(line), [line.strip()])

    def test_several_continuations_in_a_row(self):
        self.assertEqual(self.commands('color #A# red \\', '#B# blue \\', '#C# green \\', '#D# pink'),
                         ['color #A# red #B# blue #C# green #D# pink'])
        # Then the next command starts afresh.
        self.assertEqual(self.commands('color #A# red \\', '#B# blue', 'select "x" \\', '#C#'),
                         ['color #A# red #B# blue', 'select "x" #C#'])

    def test_the_line_after_a_continuation_is_command_text_even_when_it_starts_with_a_hash(self):
        # A line that starts with # is a comment, unless a continuation leads to it.
        self.assertEqual(self.commands('# comment', 'color #A# red \\', '   #B# blue', '# comment', 'select "x"'),
                         ['color #A# red #B# blue', 'select "x"'])
        self.assertEqual(self.commands('color red \\', '# not a comment'), ['color red # not a comment'])

    def test_a_comment_line_that_ends_in_a_backslash_does_not_swallow_the_next_command(self):
        self.assertEqual(self.commands('# a comment \\', 'select "x"'), ['select "x"'])
        self.assertEqual(self.commands('// a comment \\', 'select "x"'), ['select "x"'])
        self.assertEqual(self.commands('select "x" // a comment \\', 'select "y"'), ['select "x"', 'select "y"'])

    def test_comments_are_removed_from_each_physical_line_first(self):
        self.assertEqual(self.commands('color #A# red \\ // first part', '#B# blue // the rest'),
                         ['color #A# red #B# blue'])
        self.assertEqual(self.commands('color #A# red \\\t// first part', '#B# blue'), ['color #A# red #B# blue'])
        # A // that follows no whitespace is no comment, so the backslash before it is not the last character.
        self.assertEqual(self.commands('meta C://data/x.csv \\', 'more'), ['meta C://data/x.csv more'])
        # A line holding only a comment after a continuation ends the command, as a blank one does.
        self.assertEqual(self.commands('color #A# red \\', '// nothing more', 'select "x"'),
                         ['color #A# red', 'select "x"'])

    def test_the_next_line_is_joined_whatever_it_holds_so_a_blank_one_ends_the_command(self):
        self.assertEqual(self.commands('color #A# red \\', '', 'select "x"'), ['color #A# red', 'select "x"'])
        self.assertEqual(self.commands('color #A# red \\', '   \n', 'select "x"'), ['color #A# red', 'select "x"'])
        self.assertEqual(self.commands('\\', '', 'select "x"'), ['select "x"'])

    def test_tabs_and_spaces_around_the_backslash(self):
        self.assertEqual(self.commands('color #A# red\t\\', '#B# blue'), ['color #A# red #B# blue'])
        self.assertEqual(self.commands('color #A# red \\\t \t', '#B# blue'), ['color #A# red #B# blue'])
        self.assertEqual(self.commands('color #A# red   \t \\ \t \r\n', '\t #B# blue\t\r\n'),
                         ['color #A# red #B# blue'])
        # The backslash is not the last character when text follows it.
        self.assertEqual(self.commands('color #A# red \\ more', 'select "x"'),
                         ['color #A# red \\ more', 'select "x"'])

    def test_a_lone_backslash_line_continues(self):
        self.assertEqual(self.commands('\\', 'select "x"'), ['select "x"'])
        self.assertEqual(self.commands('  \\  ', 'select "x"'), ['select "x"'])
        self.assertEqual(self.commands('color #A# red \\', '\\', '#B# blue'), ['color #A# red #B# blue'])

    def test_windows_line_endings(self):
        self.assertEqual(
            self.commands('select "one"\r\n', 'color #A# red \\\r\n', '#B# blue \\\r\n', '#C# green\r\n', '\r\n',
                          'select "two"\r\n'),
            ['select "one"', 'color #A# red #B# blue #C# green', 'select "two"'])

    def test_blank_and_comment_lines_hold_no_command(self):
        self.assertEqual(self.commands('', '  ', '# a', '// b', 'select "x"', '', '# c'), ['select "x"'])
        self.assertEqual(self.commands(), [])

    def test_a_script_that_ends_in_a_continuation_is_refused_with_the_line_named(self):
        for lines, line in ((['color #A# red \\'], 1),
                            (['select "one"', 'color #A# red \\'], 2),
                            (['color #A# red \\', '#B# blue \\'], 2),
                            (['select "one"', '\\'], 2),
                            (['select "one" \\\r\n'], 1),
                            (['select "one"', '', 'color #A# red \\ // note\n'], 3)):
            with self.subTest(lines=lines):
                with self.assertRaises(ce.ScriptContinuationError) as caught:
                    ce.script_command_lines(lines)
                # The error holds the Message, which a window shows translated; str() is the English.
                self.assertEqual(str(caught.exception), DANGLING_MESSAGE.format(line=line))
                self.assertEqual(caught.exception.args[0].values, {'line': line})
                self.assertIsInstance(caught.exception, ValueError)

    def test_a_blank_line_after_the_last_continuation_is_not_a_missing_line(self):
        self.assertEqual(self.commands('color #A# red \\', ''), ['color #A# red'])
        self.assertEqual(self.commands('color #A# red \\\n', '\n'), ['color #A# red'])


# What the continued-command scripts hold, as they are written to a file or printed by a Python script.
CONTINUED = ('select "one"\n'
             'color #BCAT# cyan #C-DAAT# pink #NC-DAAT# green \\\n'
             '    #R-ATA# red \\ // first part\n'
             '#ADCL# magenta // the rest\n'
             '# a comment\n'
             'select "two"\n')
CONTINUED_EXPECTED = ['select "one"', FIGURE_COMMAND, 'select "two"']
# The last line continues, so the script is refused as a whole.
DANGLING = 'select "one"\ncolor #A# red \\\n'


class ContinuedScriptTests(RunPaths, unittest.TestCase):
    """A script's continued commands reach the Viewer as one command each, whichever way the script is run."""

    def write(self, name, text, encoding='utf-8'):
        """Write the lines text holds as a script: a .txt file holds them, a .py script prints them, byte for byte."""
        self.script = Path(self.directory.name) / name
        if name.endswith('.py'):
            self.script.write_text(f'import sys\nsys.stdout.buffer.write({text.encode()!r})\n', encoding='utf-8')
        else:
            self.script.write_bytes(text.encode(encoding))

    def test_a_continued_command_runs_as_one_command_typed_and_requested(self):
        for name in ('commands.txt', 'commands.py'):
            self.write(name, CONTINUED)
            for path, run in (('typed', lambda: self.typed(quick=True)), ('requested', self.requested)):
                with self.subTest(script=name, path=path):
                    commands, diagnostics = run()
                    self.assertEqual(commands, CONTINUED_EXPECTED, diagnostics)

    def test_a_windows_script_continues_too(self):
        crlf = CONTINUED.replace('\n', '\r\n')
        for encoding in ('utf-8', 'utf-16'):
            self.write('commands.txt', crlf, encoding)
            for path, run in (('typed', lambda: self.typed(quick=True)), ('requested', self.requested)):
                with self.subTest(encoding=encoding, path=path):
                    commands, diagnostics = run()
                    self.assertEqual(commands, CONTINUED_EXPECTED, diagnostics)
        # A Python script prints its own line endings; the Viewer reads them as text.
        self.write('commands.py', crlf)
        for path, run in (('typed', lambda: self.typed(quick=True)), ('requested', self.requested)):
            with self.subTest(script='commands.py', path=path):
                commands, diagnostics = run()
                self.assertEqual(commands, CONTINUED_EXPECTED, diagnostics)

    def test_the_lines_are_joined_before_a_nested_run_line_is_dropped(self):
        # `run` and `stop` are one command, a run command, when joined; `select "two" run` is no run command.
        text = 'select "one"\nrun \\\nstop\nselect "two" \\\nrun\n'
        expected = ['select "one"', 'select "two" run']
        for name in ('commands.txt', 'commands.py'):
            self.write(name, text)
            for path, run in (('typed', lambda: self.typed(quick=True)), ('requested', self.requested)):
                with self.subTest(script=name, path=path):
                    commands, diagnostics = run()
                    self.assertEqual(commands, expected, diagnostics)

    def test_a_typed_script_whose_last_line_continues_is_refused_before_any_command_runs(self):
        message = DANGLING_MESSAGE.format(line=2)
        for name in ('commands.txt', 'commands.py'):
            self.write(name, DANGLING)
            with self.subTest(script=name), \
                    mock.patch.object(ce, 'command_failed', wraps=ce.command_failed) as failed, \
                    mock.patch.object(ce, 'command_succeeded', wraps=ce.command_succeeded) as succeeded:
                commands, printed = self.typed(quick=True)

                # Not even the command before the last line ran.
                self.assertEqual(commands, [])
                self.assertEqual(str(failed.call_args.args[1]), message)
                self.assertIn(message, printed)
                self.assertEqual(self.viewer.console_text.text, message)
                self.assertFalse(any('Batch' in str(call.args[1]) for call in succeeded.call_args_list))
                self.assertNotIn('Batch', printed)

    def test_a_requested_script_whose_last_line_continues_fails_before_any_command_runs(self):
        message = DANGLING_MESSAGE.format(line=2)
        for name in ('commands.txt', 'commands.py'):
            self.write(name, DANGLING)
            with self.subTest(script=name):
                children, (result, diagnostics) = self.requested()

                self.assertEqual(children, [])
                self.assertEqual(result['status'], 'failed', result)
                # A .txt file's error is the command's own message; a script's is its job's, in the result.
                reported = diagnostics + ' '.join(m['text'] for c in result['commands'] for m in c['messages']) \
                    + str(result.get('jobs', ''))
                self.assertIn(message, reported)

    def test_the_tracker_joins_a_python_scripts_printed_lines_before_it_filters_them(self):
        # Viewer_Worker_Tracking's own step, whatever command starts the script: it joins, and then asks
        # keep_command about each joined command.
        from Viewer_Command_Portal import ExecutionContext
        from Viewer_Worker_Tracking import ScriptTracker
        self.write('commands.py', 'select "one"\nrun \\\nstop\n' + '\n'.join(FIGURE_LINES) + '\n')
        no_dialog = mock.patch.object(QtWidgets.QFileDialog, 'getOpenFileName', side_effect=AssertionError('no dialog'))
        for keep, expected in ((lambda line: line != 'run stop', ['select "one"', FIGURE_COMMAND]),
                               (None, ['select "one"', 'run stop', FIGURE_COMMAND])):
            with self.subTest(keep=keep is not None), no_dialog, redirect_stdout(io.StringIO()):
                request_id = self.portal.submit(str(len(self.portal.requests)), 'run')['request_id']
                record = self.portal.requests[request_id]['commands'][0]
                record.update(dispatched=True, outcome='succeeded', status='running')
                ScriptTracker(ExecutionContext(self.portal, request_id, record), str(self.script), keep_command=keep)
                deadline = time.monotonic() + 30
                while not record['children'] and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(.005)
                self.assertEqual([child['command'] for child in record['children']], expected)
                self.finish(request_id)

    def test_the_tracker_refuses_a_script_that_ends_in_a_continuation(self):
        from Viewer_Command_Portal import ExecutionContext
        from Viewer_Worker_Tracking import ScriptTracker
        self.write('commands.py', DANGLING)
        with redirect_stdout(io.StringIO()):
            request_id = self.portal.submit(str(len(self.portal.requests)), 'run')['request_id']
            record = self.portal.requests[request_id]['commands'][0]
            record.update(dispatched=True, outcome='succeeded', status='running')
            ScriptTracker(ExecutionContext(self.portal, request_id, record), str(self.script))
            deadline = time.monotonic() + 30
            while not record['jobs'] or record['jobs'][0]['status'] in ('queued', 'running'):
                if time.monotonic() > deadline:
                    self.fail('The script was still running')
                self.app.processEvents()
                time.sleep(.005)
        job = record['jobs'][0]
        self.assertEqual((job['status'], record['children']), ('failed', []))
        self.assertEqual(job['message'], DANGLING_MESSAGE.format(line=2))
        self.finish(request_id)

    def test_the_translated_refusal_names_the_line(self):
        # The error holds a Message, so the console line shows it in the language shown.
        with self.assertRaises(ce.ScriptContinuationError) as caught:
            ce.script_command_lines(['select "one"'] * 6 + ['color #A# red \\'])
        previous = Localization.set_translator(
            lambda template, n: '第{line}行以反斜杠结尾' if template == DANGLING_MESSAGE else template)
        self.addCleanup(Localization.set_translator, previous)
        self.assertEqual(display_text(caught.exception), '第7行以反斜杠结尾')
        self.assertEqual(str(caught.exception), DANGLING_MESSAGE.format(line=7))

    def test_a_joined_command_counts_once_toward_the_limit(self):
        pair = 'select "one" \\\n#A#\n'
        for pairs, expected in (
                (1000, 'Batch execution completed: 1000 commands run.'),
                (1001, 'Error: The script holds 1001 commands, more than the 1000 a script may run. None were run.')):
            with self.subTest(commands=pairs):
                self.write('commands.txt', pair * pairs)
                commands, printed = self.typed(quick=True)
                self.assertEqual(len(commands), pairs if pairs <= 1000 else 0)
                self.assertIn(expected, printed)


class TypedRunOutcomeTests(RunPaths, unittest.TestCase):
    """Typed, run stops at the first line that fails, and says which it was."""

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

    def test_a_failed_line_stops_the_script_and_names_the_line(self):
        for lines, dispatched, message in (
                # No command is named bogus; reset rejects its target.
                (('reset hide', 'bogus', 'reset colors', 'reset sizes'), ['reset hide', 'bogus'],
                 "Batch stopped at command 2 of 4: 'bogus' failed. Not run: the remaining 2 commands."),
                (('reset bogus', 'reset hide', 'bogus'), ['reset bogus'],
                 "Batch stopped at command 1 of 3: 'reset bogus' failed. Not run: the remaining 2 commands."),
                (('reset hide', 'bogus', 'reset colors'), ['reset hide', 'bogus'],
                 "Batch stopped at command 2 of 3: 'bogus' failed. Not run: the remaining 1 command."),
                # The last line leaves nothing out.
                (('reset hide', 'bogus'), ['reset hide', 'bogus'],
                 "Batch stopped at command 2 of 2: 'bogus' failed."),
                (('bogus',), ['bogus'], "Batch stopped at command 1 of 1: 'bogus' failed.")):
            with self.subTest(lines=lines):
                commands, printed, failed, succeeded = self.run_script(*lines)

                # The lines after the failure are not dispatched.
                self.assertEqual(commands, dispatched)
                self.assertEqual(str(failed.call_args.args[1]), message)
                self.assertIn(message, printed)
                self.assertEqual(self.viewer.console_text.text, message)
                self.assertNotIn('Batch execution', printed)
                self.assertFalse(any('Batch' in str(call.args[1]) for call in succeeded.call_args_list))

    def test_the_commands_a_python_script_prints_stop_at_the_first_failure_too(self):
        self.script = Path(self.directory.name) / 'commands.py'
        self.script.write_text("print('reset hide')\nprint('bogus')\nprint('reset colors')\n", encoding='utf-8')

        commands, printed = self.typed()

        self.assertEqual(commands, ['reset hide', 'bogus'])
        self.assertIn("Batch stopped at command 2 of 3: 'bogus' failed. Not run: the remaining 1 command.", printed)

    def test_comments_and_nested_run_lines_do_not_count_as_commands(self):
        commands, printed, failed, _ = self.run_script(
            '# a comment', 'run', 'reset hide // note', 'RUN again', 'bogus', 'reset colors')

        self.assertEqual(commands, ['reset hide', 'bogus'])
        self.assertEqual(str(failed.call_args.args[1]),
                         "Batch stopped at command 2 of 3: 'bogus' failed. Not run: the remaining 1 command.")
        self.assertEqual(printed.count("Recursive 'run' command in script ignored"), 2)

    def test_a_cancelled_line_stops_the_script_too(self):
        # As a cancelled command ends a request sent through the command portal.
        def dispatch(viewer, line, **options):
            (ce.command_cancelled if line == 'cancel' else ce.command_succeeded)(viewer, line)
        run_module = importlib.import_module('commands.run')
        with mock.patch.object(ce, '_dispatch_user_command', side_effect=dispatch) as dispatched, \
                mock.patch.object(ce, 'command_cancelled', wraps=ce.command_cancelled) as cancelled, \
                redirect_stdout(io.StringIO()):
            run_module.run_script_lines(self.viewer, ['one', 'cancel', 'three'])

        self.assertEqual([call.args[1] for call in dispatched.call_args_list], ['one', 'cancel'])
        self.assertEqual(str(cancelled.call_args.args[1]),
                         "Batch stopped at command 2 of 3: 'cancel' was cancelled. Not run: the remaining 1 command.")

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


class CommandLimitTests(RunPaths, unittest.TestCase):
    """A script holds at most 1000 commands, typed or requested; a longer one is refused before any command runs."""

    # Comment lines and nested run lines are not commands, so they do not count.
    def lines(self, count):
        return ['# a comment', 'run'] + ['select "one"'] * count

    def write(self, name, count):
        self.script = Path(self.directory.name) / name
        lines = self.lines(count)
        if name.endswith('.py'):
            self.script.write_text(''.join(f'print({line!r})\n' for line in lines), encoding='utf-8')
        else:
            self.script.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def test_typed_scripts_of_1000_commands_run_and_longer_ones_are_refused(self):
        for name in ('commands.txt', 'commands.py'):
            for count, message in (
                    (1000, 'Batch execution completed: 1000 commands run.'),
                    (1001, 'Error: The script holds 1001 commands, more than the 1000 a script may run. None were run.'),
                    (5000, 'Error: The script holds 5000 commands, more than the 1000 a script may run. None were run.')):
                with self.subTest(script=name, commands=count):
                    self.write(name, count)
                    with mock.patch.object(ce, 'command_failed', wraps=ce.command_failed) as failed:
                        commands, printed = self.typed(quick=True)

                    self.assertEqual(len(commands), count if count <= 1000 else 0)
                    self.assertIn(message, printed)
                    self.assertEqual(self.viewer.console_text.text, message)
                    self.assertEqual(failed.called, count > 1000)

    def test_requested_scripts_of_more_than_1000_commands_fail_before_any_runs(self):
        for name in ('commands.txt', 'commands.py'):
            with self.subTest(script=name):
                self.write(name, 1001)
                children, (result, diagnostics) = self.requested()

                self.assertEqual(children, [])
                self.assertEqual(result['status'], 'failed', result)
                # A .txt file's error is the command's own message; a script's is its job's, in stderr.
                reported = diagnostics + ' '.join(m['text'] for c in result['commands'] for m in c['messages'])
                self.assertIn('A command script may emit at most 1000', reported)


class BackgroundScriptTests(RunPaths, unittest.TestCase):
    """A typed Python script runs in the background; its commands run, on the GUI thread, when it ends."""

    def setUp(self):
        super().setUp()
        self.go = self.script.parent / 'go.txt'
        self.script.write_text(WAITING_SCRIPT, encoding='utf-8')
        self.printed = io.StringIO()
        self.threads = []
        real = ce._dispatch_user_command

        def dispatch(viewer, command, *args, **options):
            self.threads.append(threading.get_ident())
            return real(viewer, command, *args, **options)

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(redirect_stdout(self.printed))
        self.dispatched = stack.enter_context(mock.patch.object(ce, '_dispatch_user_command', side_effect=dispatch))
        stack.enter_context(self.choose_script())
        # The script's folder cannot be removed while the script still runs in it.
        self.addCleanup(self.release_and_wait)

    def release_and_wait(self):
        self.go.write_text('go')
        self.wait_for_script()

    def commands(self):
        return [call.args[1] for call in self.dispatched.call_args_list]

    def type_run(self):
        self.viewer.process_command('run', record_history=False)

    def watch_script_end(self):
        """Patch subprocess.Popen to set the returned event when the script's process has ended."""
        ended = threading.Event()

        class Watched(subprocess.Popen):
            def communicate(self, *args, **options):
                try:
                    return super().communicate(*args, **options)
                finally:
                    ended.set()

        patcher = mock.patch('subprocess.Popen', Watched)
        patcher.start()
        self.addCleanup(patcher.stop)
        return ended

    def pump(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)

    def test_run_returns_at_once_and_the_commands_run_when_the_script_ends(self):
        self.type_run()

        # The script is still waiting for go.txt: run has returned and said so, and nothing else has run.
        self.assertIsNotNone(self.viewer.command_script_run)
        self.assertEqual(self.commands(), ['run'])
        self.assertEqual(self.viewer.console_text.text, 'Executing Python script...')
        self.assertFalse(getattr(self.viewer, '_command_dispatch_active', False))
        self.assertEqual(self.viewer.selected_indices, [])

        self.go.write_text('go')
        self.wait_for_script()

        self.assertEqual(self.commands(), ['run', 'select "one"'])
        self.assertEqual(self.viewer.selected_indices, [0])
        self.assertEqual(self.viewer.console_text.text, 'Batch execution completed: 1 command run.')
        self.assertFalse(self.viewer._command_dispatch_active)
        # Everything was dispatched on this thread, the GUI thread.
        self.assertEqual(set(self.threads), {threading.get_ident()})

    def test_other_commands_work_while_the_script_runs(self):
        self.type_run()

        self.viewer.process_command('select "two"', record_history=False)
        self.assertEqual(self.viewer.selected_indices, [1])
        # A .txt script is no background script: it runs at once.
        text = Path(self.directory.name) / 'more.txt'
        text.write_text('select "three"\n', encoding='utf-8')
        with mock.patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=(str(text), '')):
            self.type_run()
        self.assertEqual(self.viewer.selected_indices, [2])
        self.assertIsNotNone(self.viewer.command_script_run)

        self.release_and_wait()

        # The script's commands run when it ends, whatever the Viewer holds then.
        self.assertEqual(self.viewer.selected_indices, [0])

    def test_a_second_script_is_refused_while_one_runs(self):
        self.type_run()
        first = self.viewer.command_script_run

        with mock.patch.object(ce, 'command_failed', wraps=ce.command_failed) as failed, \
                mock.patch('subprocess.Popen') as run:
            self.type_run()

        run.assert_not_called()
        message = 'A Python script is already running; wait for it to finish before running another.'
        self.assertEqual(str(failed.call_args.args[1]), message)
        self.assertEqual(self.viewer.console_text.text, message)
        self.assertIs(self.viewer.command_script_run, first)

        self.release_and_wait()

        # Its commands ran once, and a run is accepted again.
        self.assertEqual(self.commands(), ['run', 'run', 'select "one"'])
        with mock.patch('subprocess.Popen', wraps=subprocess.Popen) as run:
            self.type_run()
            self.wait_for_script()
        run.assert_called_once()
        self.assertEqual(self.commands()[3:], ['run', 'select "one"'])

    def test_a_script_that_fails_reports_it_and_runs_nothing(self):
        self.script.write_text(
            "import sys\nprint('select \"one\"')\nsys.stderr.write('it broke\\n')\nsys.exit(3)\n", encoding='utf-8')
        with mock.patch.object(ce, 'command_failed', wraps=ce.command_failed) as failed:
            self.type_run()
            self.wait_for_script()

        self.assertEqual(self.commands(), ['run'])
        self.assertEqual(str(failed.call_args.args[1]), 'Error: Python script failed (exit code 3):\nit broke')
        self.assertEqual(self.viewer.console_text.text, 'Error: Python script failed (exit code 3):')
        self.assertIn('it broke', self.printed.getvalue())
        self.assertEqual(self.viewer.selected_indices, [])

    def test_a_script_that_cannot_start_reports_it(self):
        with mock.patch('subprocess.Popen', side_effect=OSError('no interpreter')), \
                mock.patch.object(ce, 'command_failed', wraps=ce.command_failed) as failed:
            self.type_run()
            self.wait_for_script()

        self.assertEqual(self.commands(), ['run'])
        self.assertEqual(str(failed.call_args.args[1]), 'Error reading/executing command file: no interpreter')

    def test_the_commands_wait_for_a_command_that_is_mid_dispatch(self):
        ended = self.watch_script_end()
        self.type_run()
        # Another command is mid-dispatch, and its event processing delivers the script's result.
        self.viewer._command_dispatch_active = True
        self.go.write_text('go')
        self.assertTrue(ended.wait(30))
        self.pump(.3)

        self.assertEqual(self.commands(), ['run'])
        self.assertIsNotNone(self.viewer.command_script_run)

        self.viewer._command_dispatch_active = False
        self.wait_for_script()

        self.assertEqual(self.commands(), ['run', 'select "one"'])

    def test_a_closed_viewer_is_not_touched(self):
        ended = self.watch_script_end()
        self.type_run()
        self.viewer.command_script_run.abandon()  # What the canvas's close event does.
        self.go.write_text('go')
        self.assertTrue(ended.wait(30))
        self.pump(.3)

        self.assertEqual(self.commands(), ['run'])
        self.assertEqual(self.viewer.console_text.text, 'Executing Python script...')
        self.assertIsNone(self.viewer.command_script_run)

    def test_the_canvas_close_event_abandons_the_run(self):
        from vispy.util.event import EmitterGroup, Event
        self.viewer.canvas = SimpleNamespace(native=None, events=EmitterGroup(source=None, close=Event))
        self.type_run()

        self.viewer.canvas.events.close()

        self.assertTrue(self.viewer.command_script_run._abandoned)

    def test_a_viewer_that_is_gone_is_not_touched(self):
        from Viewer_Worker_Tracking import TypedScriptRun
        ended = self.watch_script_end()
        finished = mock.Mock()
        viewer = Viewer(self.directory.name)
        run = TypedScriptRun(viewer, str(self.script), finished)
        del viewer
        gc.collect()
        self.go.write_text('go')
        self.assertTrue(ended.wait(30))

        self.pump(.3)

        finished.assert_not_called()
        self.assertIsNotNone(run)


class StopScriptTests(RunPaths, unittest.TestCase):
    """`run stop`, and closing the Viewer, end a typed Python script and every
    process it started; none of its commands run."""

    def setUp(self):
        super().setUp()
        self.printed = io.StringIO()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(redirect_stdout(self.printed))
        self.dispatched = stack.enter_context(
            mock.patch.object(ce, '_dispatch_user_command', wraps=ce._dispatch_user_command))
        stack.enter_context(self.choose_script())

    def commands(self):
        return [call.args[1] for call in self.dispatched.call_args_list]

    def start_tree_script(self):
        """Type run on TREE_SCRIPT; return the script's and its own process, once both run."""
        self.script.write_text(TREE_SCRIPT, encoding='utf-8')
        pids = self.script.parent / 'pids.txt'
        self.viewer.process_command('run', record_history=False)
        deadline = time.monotonic() + 30
        while not pids.exists():
            if time.monotonic() >= deadline:
                self.fail('The script did not start its process within 30 s')
            time.sleep(.01)
        processes = [psutil.Process(int(pid)) for pid in pids.read_text().split()]
        for process in processes:
            self.addCleanup(self.kill_quietly, process)
        # The script's folder cannot be removed while the script still runs in it.
        self.addCleanup(self.wait_for_script)
        return processes

    @staticmethod
    def kill_quietly(process):
        try:
            process.kill()
        except psutil.Error:
            pass

    def assert_ended(self, processes):
        _gone, alive = psutil.wait_procs(processes, timeout=10)
        self.assertEqual(alive, [])

    def test_run_stop_ends_the_script_and_the_process_it_started(self):
        processes = self.start_tree_script()

        self.viewer.process_command('run stop', record_history=False)

        self.assertEqual(self.viewer.console_text.text, 'Stopped 1 Python script.')
        self.assert_ended(processes)
        self.wait_for_script()
        self.assertEqual(self.commands(), ['run', 'run stop'])
        # The stopped script is not reported as a failure once it has ended.
        self.assertEqual(self.viewer.console_text.text, 'Stopped 1 Python script.')
        self.assertNotIn('Python script failed', self.printed.getvalue())
        # A new script may start.
        self.assertIsNone(self.viewer.command_script_run)

    def test_run_stop_with_no_script_running_says_so(self):
        with mock.patch.object(ce, 'command_succeeded', wraps=ce.command_succeeded) as succeeded:
            self.viewer.process_command('run stop', record_history=False)

        self.assertEqual(str(succeeded.call_args.args[1]), 'No Python script is running.')
        self.assertEqual(self.viewer.console_text.text, 'No Python script is running.')

    def test_closing_the_viewer_ends_the_script_and_the_process_it_started(self):
        from vispy.util.event import EmitterGroup, Event
        self.viewer.canvas = SimpleNamespace(native=None, events=EmitterGroup(source=None, close=Event))
        processes = self.start_tree_script()

        self.viewer.canvas.events.close()

        self.assert_ended(processes)
        self.wait_for_script()
        self.assertEqual(self.commands(), ['run'])

    def test_quitting_the_application_stops_every_script(self):
        import Viewer_Worker_Tracking as tracking
        application = SimpleNamespace(aboutToQuit=mock.Mock())
        self.script.write_text("print('select \"one\"')\n", encoding='utf-8')
        results = queue.Queue()
        with mock.patch.object(tracking, '_quit_hooked', False), \
                mock.patch.object(tracking.QtCore, 'QCoreApplication',
                                  SimpleNamespace(instance=lambda: application)):
            tracking.start_command_script(str(self.script), SimpleNamespace(emit=results.put))
            tracking.start_command_script(str(self.script), SimpleNamespace(emit=results.put))
        application.aboutToQuit.connect.assert_called_once_with(tracking.stop_command_scripts)
        for _ in range(2):
            self.assertEqual(results.get(timeout=30).returncode, 0)


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
