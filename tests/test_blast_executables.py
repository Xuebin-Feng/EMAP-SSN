# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""BLAST_Executables: the one lookup for blastp and makeblastdb (a BLASTP
Directory that is set is used or refused, a blank one searches the PATH and the
standard folders), the usage-reporting environment every BLAST+ process gets,
and the version query."""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utilities import BLAST_Executables as blast  # noqa: E402


def install(folder, names=blast.PROGRAMS, windows=None):
    """Executable stand-ins for BLAST+ programs in folder, named as on this host (or as on Windows or not); their paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if (blast.is_windows() if windows is None else windows) else ""
    paths = []
    for name in names:
        path = folder / (name + suffix)
        path.write_bytes(b"")
        path.chmod(0o755)
        paths.append(str(path))
    return paths


class SettingTests(unittest.TestCase):
    """A BLASTP Directory that is set must hold both programs; nothing else is tried."""

    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        # A BLAST+ the machine has must not decide any of these.
        self.enterContext(mock.patch.object(blast, "find_automatic", side_effect=AssertionError("fell through")))

    def test_a_folder_with_both_programs_is_used(self):
        blastp, makeblastdb = install(self.folder)
        found = blast.find_executables(str(self.folder))
        self.assertEqual(found, (blastp, makeblastdb, "BLASTP Directory"))
        self.assertEqual(found.directory, str(self.folder))

    def test_the_setting_is_read_when_called_not_when_imported(self):
        # The runner loads its settings after its imports: a folder named later is the one used.
        first, second = self.folder / "first", self.folder / "second"
        install(first)
        install(second)
        self.assertEqual(blast.find_executables(str(first)).directory, str(first))
        self.assertEqual(blast.find_executables(str(second)).directory, str(second))

    def test_a_folder_that_does_not_exist_is_named(self):
        missing = self.folder / "nowhere"
        with self.assertRaises(blast.BlastError) as caught:
            blast.find_executables(str(missing))
        self.assertIn(f"The BLASTP Directory '{missing}' is not a folder that exists", str(caught.exception))

    def test_a_missing_program_is_named_and_nothing_else_is_tried(self):
        install(self.folder, ("blastp",))
        with self.assertRaises(blast.BlastError) as caught:
            blast.find_executables(str(self.folder))
        message = str(caught.exception)
        self.assertIn(f"The BLASTP Directory '{self.folder}' does not contain 'makeblastdb", message)
        self.assertNotIn("'blastp", message)

    def test_an_empty_folder_names_both_programs(self):
        with self.assertRaises(blast.BlastError) as caught:
            blast.find_executables(str(self.folder))
        message = str(caught.exception)
        self.assertIn(f"'{blast.program_file('blastp')}' and '{blast.program_file('makeblastdb')}'", message)

    def test_an_install_root_points_at_its_bin_folder(self):
        install(self.folder / "bin")
        with self.assertRaises(blast.BlastError) as caught:
            blast.find_executables(str(self.folder))
        self.assertIn(f"use '{self.folder / 'bin'}'", str(caught.exception))

    @unittest.skipIf(os.name == "nt", "Windows has no execute bit")
    def test_a_program_that_is_not_executable_is_refused(self):
        _, makeblastdb = install(self.folder)
        os.chmod(makeblastdb, stat.S_IRUSR | stat.S_IWUSR)
        with self.assertRaises(blast.BlastError) as caught:
            blast.find_executables(str(self.folder))
        self.assertIn("'makeblastdb' in the BLASTP Directory", str(caught.exception))
        self.assertIn("not executable", str(caught.exception))

    def test_a_setting_that_names_a_file_is_refused(self):
        file = self.folder / "blastp"
        file.write_bytes(b"")
        with self.assertRaises(blast.BlastError):
            blast.find_executables(str(file))

    def test_the_unix_folders_are_the_ones_the_tool_listed_before(self):
        self.assertEqual(
            blast.UNIX_FOLDERS,
            ("/usr/local/ncbi/blast/bin", "/usr/local/bin", "/usr/bin", "/opt/homebrew/bin"),
        )


class AutomaticSearchTests(unittest.TestCase):
    """A blank BLASTP Directory: the PATH, then the standard folders of the platform."""

    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = {}  # what shutil.which finds, by program name
        self.enterContext(mock.patch.object(blast.shutil, "which", side_effect=lambda name: self.path.get(name)))
        self.enterContext(mock.patch.object(blast, "windows_ncbi_root", return_value=str(self.folder / "NCBI")))
        self.enterContext(mock.patch.object(blast, "UNIX_FOLDERS", ()))

    def test_blank_values_search_automatically(self):
        for blank in ("", "   ", None):
            with self.subTest(blank=blank), self.assertRaises(blast.BlastError) as caught:
                blast.find_executables(blank)
            self.assertIn("was not found", str(caught.exception))
        blastp, makeblastdb = install(self.folder / "p")
        self.path = {"blastp": blastp, "makeblastdb": makeblastdb}
        self.assertEqual(blast.find_executables("  ").source, "PATH")

    def test_the_path_comes_first(self):
        blastp, makeblastdb = install(self.folder / "p")
        self.path = {"blastp": blastp, "makeblastdb": makeblastdb}
        with mock.patch.object(blast, "windows_install_folders", side_effect=AssertionError("searched")):
            found = blast.find_executables("")
        self.assertEqual(found, (os.path.abspath(blastp), os.path.abspath(makeblastdb), "PATH"))

    def test_a_relative_path_entry_is_made_absolute_for_the_workers_that_run_elsewhere(self):
        self.path = {"blastp": os.path.join("bin", "blastp"), "makeblastdb": os.path.join("bin", "makeblastdb")}
        found = blast.find_executables("")
        self.assertEqual(found.blastp, os.path.abspath(os.path.join("bin", "blastp")))
        self.assertEqual(found.makeblastdb, os.path.abspath(os.path.join("bin", "makeblastdb")))

    def test_makeblastdb_beside_blastp_is_preferred_to_another_one_on_the_path(self):
        first, second = install(self.folder / "a"), install(self.folder / "b")
        self.path = {"blastp": first[0], "makeblastdb": second[1]}
        self.assertEqual(blast.find_executables("").makeblastdb, os.path.abspath(first[1]))

    def test_blastp_without_makeblastdb_is_not_usable_and_the_error_says_so(self):
        (blastp,) = install(self.folder / "a", ("blastp",))
        self.path = {"blastp": blastp}
        with self.assertRaises(blast.BlastError) as caught:
            blast.find_executables("")
        self.assertIn(f"blastp is at '{blastp}'", str(caught.exception))
        self.assertIn("makeblastdb is not beside it", str(caught.exception))

    def test_an_incomplete_path_entry_gives_way_to_a_standard_folder(self):
        (blastp,) = install(self.folder / "a", ("blastp",))
        self.path = {"blastp": blastp}
        usr = self.folder / "usr"
        standard = install(usr, windows=False)
        with mock.patch.object(blast, "is_windows", return_value=False), \
                mock.patch.object(blast, "UNIX_FOLDERS", (str(usr),)):
            found = blast.find_executables("")
        self.assertEqual(found.blastp, standard[0])

    def test_windows_takes_the_newest_version_by_number_not_by_name(self):
        root = self.folder / "NCBI"
        for name in ("blast-2.9.0+", "blast-2.17.0+", "blast-2.10.1+", "blast-2.2.31+", "scratch"):
            install(root / name / "bin", windows=True)
        with mock.patch.object(blast, "is_windows", return_value=True):
            folders = blast.windows_install_folders(str(root))
            found = blast.find_executables("")
        # Sorted as text, 2.9.0 would come first and 2.17.0 third.
        self.assertEqual(
            [Path(folder).parent.name for folder in folders],
            ["blast-2.17.0+", "blast-2.10.1+", "blast-2.9.0+", "blast-2.2.31+", "scratch"],
        )
        self.assertEqual(found.directory, str(root / "blast-2.17.0+" / "bin"))
        self.assertEqual(found.source, f"{root / 'blast-2.17.0+' / 'bin'} (standard install folder)")

    def test_windows_takes_2_10_over_2_9_even_though_text_order_says_otherwise(self):
        root = self.folder / "NCBI"
        for name in ("blast-2.9.0+", "blast-2.10.0+"):
            install(root / name / "bin", windows=True)
        with mock.patch.object(blast, "is_windows", return_value=True):
            self.assertEqual(blast.find_executables("").directory, str(root / "blast-2.10.0+" / "bin"))

    def test_windows_skips_a_folder_that_lacks_a_program(self):
        root = self.folder / "NCBI"
        install(root / "blast-2.17.0+" / "bin", ("blastp",), windows=True)
        install(root / "blast-2.9.0+" / "bin", windows=True)
        with mock.patch.object(blast, "is_windows", return_value=True):
            self.assertEqual(blast.find_executables("").directory, str(root / "blast-2.9.0+" / "bin"))

    def test_windows_without_the_installer_folder_finds_nothing(self):
        with mock.patch.object(blast, "is_windows", return_value=True):
            self.assertIsNone(blast.find_automatic())
            with self.assertRaises(blast.BlastError) as caught:
                blast.find_executables("")
        self.assertIn("standard NCBI install folder", str(caught.exception))
        self.assertIn("BLASTP Directory", str(caught.exception))

    def test_unix_folders_are_tried_in_the_listed_order(self):
        first, second = self.folder / "usr-local", self.folder / "usr"
        install(second, windows=False)
        listed = (str(self.folder / "none"), str(first), str(second))
        with mock.patch.object(blast, "is_windows", return_value=False), \
                mock.patch.object(blast, "UNIX_FOLDERS", listed):
            self.assertEqual(blast.find_executables("").directory, str(second))
            install(first, windows=False)
            found = blast.find_executables("")
        self.assertEqual(found.directory, str(first))
        self.assertEqual(found.source, f"{first} (standard install folder)")

    def test_the_placeholder_says_what_the_search_would_use_or_that_it_found_nothing(self):
        usr = self.folder / "usr"
        with mock.patch.object(blast, "is_windows", return_value=False):
            self.assertIsNone(blast.find_automatic())
            self.assertEqual(blast.describe_automatic(), "BLAST+ not found")
            install(usr, windows=False)
            with mock.patch.object(blast, "UNIX_FOLDERS", (str(usr),)):
                self.assertEqual(blast.describe_automatic(), f"{usr} (standard install folder)")
            blastp, makeblastdb = install(self.folder / "p", windows=False)
            self.path = {"blastp": blastp, "makeblastdb": makeblastdb}
            self.assertEqual(blast.describe_automatic(), f"{os.path.dirname(os.path.abspath(blastp))} (PATH)")

    def test_the_search_never_runs_blast(self):
        blastp, makeblastdb = install(self.folder / "p")
        self.path = {"blastp": blastp, "makeblastdb": makeblastdb}
        with mock.patch.object(subprocess, "run", side_effect=AssertionError("ran")), \
                mock.patch.object(subprocess, "Popen", side_effect=AssertionError("ran")):
            blast.find_automatic()
            blast.describe_automatic()


class ImportTests(unittest.TestCase):
    def test_importing_looks_for_nothing_and_runs_nothing(self):
        code = (
            "import os, shutil, subprocess, sys\n"
            f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
            "def refuse(*args, **kwargs): raise SystemExit('touched BLAST on import')\n"
            "shutil.which = refuse\n"
            "subprocess.run = subprocess.Popen = refuse\n"
            "os.listdir = os.scandir = refuse\n"
            "from utilities import BLAST_Executables\n"
        )
        completed = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)


class EnvironmentTests(unittest.TestCase):
    def test_usage_reporting_is_turned_off_and_the_rest_is_kept(self):
        environment = blast.blast_environment({"PATH": "x", "HOME": "y"})
        self.assertEqual(environment, {"PATH": "x", "HOME": "y", "BLAST_USAGE_REPORT": "false"})

    def test_the_variables_that_take_precedence_are_never_touched(self):
        for kept in ({"DO_NOT_TRACK": "1"}, {"NCBI_USAGE_REPORT_ENABLED": "true"}, {}):
            with self.subTest(kept=kept):
                environment = blast.blast_environment(dict(kept))
                self.assertEqual({key: environment[key] for key in kept}, kept)
                self.assertEqual(set(environment) - set(kept), {"BLAST_USAGE_REPORT"})

    def test_a_users_own_blast_usage_report_is_replaced(self):
        self.assertEqual(blast.blast_environment({"BLAST_USAGE_REPORT": "true"})["BLAST_USAGE_REPORT"], "false")

    def test_the_current_environment_is_copied_not_changed(self):
        with mock.patch.dict(os.environ, {"EMAPSSN_BLAST_TEST": "1"}):
            os.environ.pop("BLAST_USAGE_REPORT", None)
            environment = blast.blast_environment()
            self.assertEqual(environment["EMAPSSN_BLAST_TEST"], "1")
            self.assertEqual(environment["BLAST_USAGE_REPORT"], "false")
            self.assertNotIn("BLAST_USAGE_REPORT", os.environ)


class VersionTests(unittest.TestCase):
    @staticmethod
    def completed(stdout="blastp: 2.17.0+\n Package: blast 2.17.0, build Jun 2 2026 12:00:00\n", stderr=""):
        return mock.Mock(returncode=0, stdout=stdout, stderr=stderr)

    def test_the_first_line_is_the_version_and_usage_reporting_is_off(self):
        with mock.patch.object(subprocess, "run", return_value=self.completed()) as run:
            self.assertEqual(blast.query_version("/x/blastp"), "blastp: 2.17.0+")
        self.assertEqual(run.call_args.args[0], ["/x/blastp", "-version"])
        self.assertEqual(run.call_args.kwargs["env"]["BLAST_USAGE_REPORT"], "false")
        self.assertEqual(run.call_args.kwargs["timeout"], 30)

    def test_a_blank_first_line_is_skipped_and_stderr_is_read_when_stdout_is_empty(self):
        with mock.patch.object(subprocess, "run", return_value=self.completed("", "\n  blastp: 2.9.0+\n")):
            self.assertEqual(blast.query_version("blastp"), "blastp: 2.9.0+")

    def test_a_program_that_cannot_start_or_says_nothing_is_a_blast_error(self):
        for failure in (OSError("not found"), subprocess.TimeoutExpired("blastp", 30),
                        subprocess.CalledProcessError(3221225781, "blastp")):
            with self.subTest(failure=type(failure).__name__), \
                    mock.patch.object(subprocess, "run", side_effect=failure):
                with self.assertRaises(blast.BlastError) as caught:
                    blast.query_version("C:/x/blastp.exe")
                self.assertIn("Unable to query BLASTP version using 'C:/x/blastp.exe'", str(caught.exception))
        with mock.patch.object(subprocess, "run", return_value=self.completed("", "")):
            with self.assertRaisesRegex(blast.BlastError, "returned no version information"):
                blast.query_version("blastp")

    def test_a_cached_version_is_asked_once_per_file_and_again_when_the_file_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            program = Path(folder) / "blastp"
            program.write_bytes(b"one")
            with mock.patch.object(subprocess, "run", return_value=self.completed()) as run:
                self.assertEqual(blast.cached_version(str(program)), "blastp: 2.17.0+")
                self.assertEqual(blast.cached_version(str(program)), "blastp: 2.17.0+")
                self.assertEqual(run.call_count, 1)
                program.write_bytes(b"a replaced program")
                blast.cached_version(str(program))
                self.assertEqual(run.call_count, 2)

    def test_a_failed_query_is_not_remembered(self):
        with tempfile.TemporaryDirectory() as folder:
            program = Path(folder) / "blastp"
            program.write_bytes(b"x")
            with mock.patch.object(subprocess, "run", side_effect=OSError("busy")):
                with self.assertRaises(blast.BlastError):
                    blast.cached_version(str(program))
            with mock.patch.object(subprocess, "run", return_value=self.completed()):
                self.assertEqual(blast.cached_version(str(program)), "blastp: 2.17.0+")


if __name__ == "__main__":
    unittest.main()
