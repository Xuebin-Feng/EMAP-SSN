"""The guard that keeps tests off the user's desktop (tests/desktop_openers.py):
each call that opens a folder, a file, a page or a console is replaced and
recorded, a test that reaches one it did not expect fails, and the replacements
are in every process that imports the tests package."""

import io
import os
import subprocess
import sys
import tempfile
import unittest
import webbrowser
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from tests import desktop_openers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from desktop import Desktop_App  # noqa: E402


class TrapsTests(unittest.TestCase):
    def test_a_page_is_not_opened_and_the_call_is_recorded(self):
        for name in ("open", "open_new", "open_new_tab"):
            with self.subTest(call=name), desktop_openers.expecting() as calls:
                self.assertTrue(getattr(webbrowser, name)("http://127.0.0.1:1/page"))
                self.assertEqual(len(calls), 1)
                self.assertIn("http://127.0.0.1:1/page", calls[0])
                self.assertIn(f"webbrowser.{name}", calls[0])

    @unittest.skipUnless(hasattr(os, "startfile"), "os.startfile exists on Windows only")
    def test_a_file_is_not_started_and_the_call_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder, desktop_openers.expecting() as calls:
            os.startfile(folder)
        self.assertEqual(len(calls), 1)
        self.assertIn("os.startfile", calls[0])

    def test_qt_does_not_open_a_url_and_the_call_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder, desktop_openers.expecting() as calls:
            self.assertTrue(QDesktopServices.openUrl(QUrl.fromLocalFile(folder)))
        self.assertEqual(len(calls), 1)
        self.assertIn("QDesktopServices.openUrl", calls[0])
        self.assertIn(QUrl.fromLocalFile(folder).toString(), calls[0])

    def test_the_file_manager_opener_reaches_nothing(self):
        # The program's own opener, through Qt and then the platform shell, ends in the traps.
        with tempfile.TemporaryDirectory() as folder, desktop_openers.expecting() as calls:
            self.assertTrue(Desktop_App.open_in_file_manager(folder))
        self.assertEqual(len(calls), 1, calls)

    def test_an_opener_program_is_not_started(self):
        for command in (["xdg-open", "x"], ["open", "x"], ["gio", "open", "x"], ["explorer", "x"],
                        ["explorer.exe", "/select,x"], [r"C:\Windows\explorer.exe", "x"], "xdg-open x"):
            with self.subTest(command=command), desktop_openers.expecting() as calls:
                with self.assertRaises(desktop_openers.DesktopOpenerError):
                    subprocess.Popen(command)
                self.assertEqual(len(calls), 1)
                self.assertIn("subprocess.Popen", calls[0])

    @unittest.skipUnless(os.name == "nt", "CREATE_NEW_CONSOLE exists on Windows only")
    def test_a_new_console_is_not_made(self):
        with desktop_openers.expecting() as calls:
            with self.assertRaises(desktop_openers.DesktopOpenerError):
                subprocess.Popen([sys.executable, "-c", "pass"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        self.assertEqual(len(calls), 1)
        self.assertIn("CREATE_NEW_CONSOLE", calls[0])

    def test_other_programs_still_start_and_are_not_recorded(self):
        with desktop_openers.expecting() as calls:
            done = subprocess.run([sys.executable, "-c", "import sys; sys.exit(3)"], check=False)
            with subprocess.Popen([sys.executable, "-c", "pass"]) as process:
                process.wait(timeout=60)
        self.assertEqual((done.returncode, process.returncode, calls), (3, 0, []))

    def test_calls_in_nested_blocks_go_to_the_innermost(self):
        with desktop_openers.expecting() as outer:
            with desktop_openers.expecting() as inner:
                webbrowser.open("http://127.0.0.1:1/inner")
            webbrowser.open("http://127.0.0.1:1/outer")
        self.assertEqual([len(inner), len(outer)], [1, 1])
        self.assertIn("/inner", inner[0])
        self.assertIn("/outer", outer[0])


class FailingTests(unittest.TestCase):
    """A test that reaches an opener fails, whatever its code did with the answer."""

    def run_nested(self, *tests):
        class Nested(unittest.TestCase):
            def test_reaches_the_browser(self):
                webbrowser.open("http://127.0.0.1:1/reached")

            def test_reaches_the_browser_and_ignores_the_answer(self):
                try:
                    webbrowser.open("http://127.0.0.1:1/ignored")
                except Exception:
                    pass

            def test_reaches_the_file_manager(self):
                Desktop_App.open_in_file_manager(tempfile.gettempdir())

            def test_expects_the_browser(self):
                with desktop_openers.expecting():
                    webbrowser.open("http://127.0.0.1:1/expected")

            def test_reaches_nothing(self):
                pass

        stream = io.StringIO()
        suite = unittest.TestSuite(Nested(name) for name in tests)
        result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
        return result, stream.getvalue()

    def test_a_test_that_reaches_an_opener_fails_and_names_it(self):
        result, output = self.run_nested("test_reaches_the_browser")
        self.assertEqual(len(result.failures), 1, output)
        self.assertEqual(result.errors, [])
        message = result.failures[0][1]
        self.assertIn("webbrowser.open('http://127.0.0.1:1/reached')", message)
        self.assertIn("test_desktop_openers.py", message, "the line that called it")
        self.assertIn("test_reaches_the_browser", message, "the test")

    def test_the_program_s_own_file_manager_opener_fails_the_test_too(self):
        result, _ = self.run_nested("test_reaches_the_file_manager")
        self.assertEqual(len(result.failures), 1)
        self.assertIn("QDesktopServices.openUrl", result.failures[0][1])
        self.assertIn("Desktop_App.py", result.failures[0][1], "the program's own line")

    def test_swallowing_the_answer_does_not_hide_the_call(self):
        result, _ = self.run_nested("test_reaches_the_browser_and_ignores_the_answer")
        self.assertEqual(len(result.failures), 1)

    def test_an_expected_call_and_a_test_that_reaches_nothing_pass(self):
        result, output = self.run_nested("test_expects_the_browser", "test_reaches_nothing")
        self.assertTrue(result.wasSuccessful(), output)
        self.assertEqual(result.testsRun, 2)
        self.assertEqual(desktop_openers.unexpected_calls(), [])

    def test_each_failure_is_charged_to_its_own_test(self):
        result, _ = self.run_nested("test_reaches_the_browser", "test_reaches_nothing",
                                    "test_reaches_the_file_manager")
        self.assertEqual(sorted(test._testMethodName for test, _ in result.failures),
                         ["test_reaches_the_browser", "test_reaches_the_file_manager"])


class EveryProcessTests(unittest.TestCase):
    """The traps belong to the tests package, so a child that imports it has them too."""

    def child(self, code):
        # The environment of a section's own child: no marker, so it sets its folders up itself.
        environment = {name: value for name, value in os.environ.items() if name != "EMAPSSN_TEST_ENVIRONMENT"}
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              cwd=PROJECT_ROOT, env=environment, timeout=120, check=False)

    def test_a_child_that_imports_the_package_has_the_traps(self):
        done = self.child(
            "import tests, webbrowser, os\n"
            "assert webbrowser.open.__module__ == 'tests.desktop_openers', webbrowser.open\n"
            "assert not hasattr(os, 'startfile') or os.startfile.__module__ == 'tests.desktop_openers'\n"
            "print('trapped')\n")
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "trapped"), done.stderr)

    def test_a_call_with_no_test_running_is_reported_when_the_process_ends(self):
        done = self.child(
            "import tests, webbrowser\n"
            "webbrowser.open('http://127.0.0.1:1/late')\n")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("Desktop openers reached while no test was running", done.stderr)
        self.assertIn("http://127.0.0.1:1/late", done.stderr)

    def test_installing_twice_changes_nothing(self):
        before = (webbrowser.open, subprocess.Popen, unittest.TestResult.stopTest, QDesktopServices.openUrl)
        desktop_openers.install()
        self.assertEqual((webbrowser.open, subprocess.Popen, unittest.TestResult.stopTest, QDesktopServices.openUrl),
                         before)


if __name__ == "__main__":
    unittest.main()
