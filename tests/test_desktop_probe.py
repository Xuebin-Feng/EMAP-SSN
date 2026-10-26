# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The desktop probe (tests/desktop_probe/): a test-only hook that loads into EMAP-SSN's own processes in a test clone, and lets an agent see, capture and operate their Qt windows with no desktop permission.

What is checked here: the gate and where the probe acts (only this checkout's entry points, and never in a test run, pip or another script), the install and uninstall of the .pth, that an inert venv costs a stat call and writes nothing, the redaction of keys, the in-process part (window events, automatic and requested captures, every action, modal dialogs that keep the probe polling, the queue that answers file dialogs, the Desktop_Open record, the probe turning itself off on a failure and when the gate goes), the command line (do, wait, step, status, timeline, sheet) and the files it reads. Nothing here opens a window: Qt runs offscreen."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = PROJECT_ROOT / "tests" / "desktop_probe"
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


boot = load_module("desktop_probe_boot_under_test", PROBE_DIR / "boot.py")
boot.load_package(str(PROBE_DIR))
from emapssn_desktop_probe import recorder, sheet, store  # noqa: E402

probe_cli = load_module("desktop_probe_cli_under_test", PROBE_DIR / "probe.py")


def clean_environment(**extra):
    """The environment an agent's shell has: no test-run marker, none of the suite's SSN_ variables."""
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith("SSN_") and name != boot.TEST_ENVIRONMENT}
    environment.update(QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1", **extra)
    return environment


def make_checkout(root: Path) -> Path:
    """A checkout that has only what the probe looks at: its own folder, a venv's site-packages and src/."""
    shutil.copytree(PROBE_DIR, root / "tests" / "desktop_probe", ignore=shutil.ignore_patterns("__pycache__"))
    (root / ".venv" / "Lib" / "site-packages").mkdir(parents=True)
    (root / "src" / "desktop").mkdir(parents=True)
    return root


def run_json(arguments, **kwargs):
    """Run probe_cli.main(arguments) and return (exit code, what it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
        try:
            code = probe_cli.main(arguments)
        except SystemExit as error:
            code = error.code
    return code, buffer.getvalue()


# ---------------------------------------------------------------------------------------------
# The gate, the redaction and the files
# ---------------------------------------------------------------------------------------------

class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_the_gate_is_a_file_whose_presence_is_the_switch(self):
        self.assertIsNone(store.read_gate(self.root))
        store.write_gate(self.root, self.root / "elsewhere", settle_ms=50)
        gate = store.read_gate(self.root)
        self.assertEqual(Path(gate["out"]), (self.root / "elsewhere").resolve())
        self.assertEqual(gate["settle_ms"], 50)
        self.assertTrue(store.remove_gate(self.root))
        self.assertIsNone(store.read_gate(self.root))
        self.assertFalse(store.remove_gate(self.root))

    def test_an_empty_or_broken_gate_file_still_turns_the_probe_on_with_the_defaults(self):
        for text in ("", "not json", "[1]"):
            with self.subTest(text=text):
                store.gate_file(self.root).parent.mkdir(parents=True, exist_ok=True)
                store.gate_file(self.root).write_text(text, encoding="utf-8")
                gate = store.read_gate(self.root)
                self.assertEqual(Path(gate["out"]), store.default_out(self.root))
                self.assertTrue(gate["auto_capture"])

    def test_the_output_folder_is_remembered_after_off(self):
        store.write_gate(self.root, self.root / "run1")
        store.remove_gate(self.root)
        self.assertEqual(store.out_for(self.root), (self.root / "run1").resolve())
        self.assertEqual(store.out_for(self.root, self.root / "other"), (self.root / "other").resolve())

    def test_keys_tokens_and_named_secrets_are_masked(self):
        secrets = [
            "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789", "hf_abcdefghijklmnopqrstuvwxyz012345",
            "AIzaSyA-abcdefghijklmnopqrstuvwxyz01234", "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123-_xyz",
            "Bearer abcdefgh12345678", "AKIAABCDEFGHIJKLMNOP",
        ]
        for secret in secrets:
            with self.subTest(secret=secret[:12]):
                masked = store.redact_text(f"the key is {secret} and more")
                self.assertNotIn(secret, masked)
                self.assertIn(store.MASK, masked)
        for text, value in (('{"api_key": "plainvalue99"}', "plainvalue99"), ("MY_TOKEN=abcd1234", "abcd1234"),
                            ("--password=hunter22", "hunter22"), ("password: 'swordfish'", "swordfish")):
            with self.subTest(text=text):
                self.assertNotIn(value, store.redact_text(text))
        self.assertNotIn("A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7", store.redact_text("id A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7 end"))

    def test_ordinary_text_is_not_masked(self):
        for text in ("Token: paste it here", "Save the output", "C:/Users/me/Cache_Files/version_00.h5",
                     "XP_007399765.1 and " + "ACDEFGHIKLMNPQRSTVWY" * 3, "Cache layout 2 of 3", "a-long-but-plain-file-name_without_digits_in_it.txt"):
            with self.subTest(text=text):
                self.assertEqual(store.redact_text(text), text)

    FASTA_HEADERS = (
        # UniProt headers as the programs show them, the spaces replaced by underscores (a node's name in a table or a title)
        "sp|Q5ZKK2|GEMI4_CHICK_Gem-associated_protein_4_OS=Gallus_gallus_OX=9031_GN=GEMIN4_PE=2_SV=1",
        "tr|A0A1D5PGT2|A0A1D5PGT2_CHICK_Secretogranin-2_OS=Gallus_gallus_OX=9031_GN=SCG2_PE=3_SV=1",
        "sp|P49768|PSN1_HUMAN_Presenilin-1_gamma-secretase_subunit_OS=Homo_sapiens_OX=9606_GN=PSEN1_PE=1_SV=1",
        "sp|P47872|SMS_HUMAN_Secretin_receptor_OS=Homo_sapiens_OX=9606_GN=SCTR_PE=1_SV=2",
        "sp|Q9Y2X3|NOP58_HUMAN_Nucleolar_protein_58_OS=Homo_sapiens_OX=9606_GN=NOP58_PE=1_SV=1",
        # the same with its spaces, and as a filename or a label an agent may read
        ">sp|Q5ZKK2|GEMI4_CHICK Gem-associated protein 4 OS=Gallus gallus OX=9031 GN=GEMIN4 PE=2 SV=1",
        "Cluster_alignment_of_the_selected_nodes_2024_05_report_final.txt",
        "Sequence_Sets/sp|Q5ZKK2|GEMI4_CHICK_Gem-associated_protein_4_OS=Gallus_gallus",
    )

    def test_a_fasta_header_is_not_taken_for_a_key(self):
        for header in self.FASTA_HEADERS:
            with self.subTest(header=header[:50]):
                self.assertEqual(store.redact_text(header), header)
                self.assertEqual(store.redact_data({"title": header, "lines": [header]}), {"title": header, "lines": [header]})

    def test_a_key_or_a_credential_beside_a_header_or_in_a_long_name_is_still_masked(self):
        header = self.FASTA_HEADERS[0]
        key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"
        random_runs = (
            "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7",              # letters and digits mixed, 34 long
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",  # a SHA-256
            "dGhpcyBpcyBhIHNlY3JldCB0b2tlbiB2YWx1ZQ12345678",  # base64
            "Xk3_9aQ-mZ7pL2vN8wR4tY6uB1cD5eF0gH3jK",           # url-safe base64 with a join or two among long pieces
        )
        for run in random_runs:
            with self.subTest(run=run[:12]):
                for text in (f"{header} {run}", f"|{run}=", f"token {run}"):
                    masked = store.redact_text(text)
                    self.assertNotIn(run, masked, text)
                self.assertIn(header, store.redact_text(f"{header} {run}"))  # the header is left as it was
        self.assertNotIn(key, store.redact_text(f"{header} {key}"))
        for text, value in (("client_secret=abcd1234efgh", "abcd1234efgh"), ("secrets: 'swordfish99'", "swordfish99"),
                            ("MyToken=abcd1234", "abcd1234"), ('{"accessToken": "plainvalue99"}', "plainvalue99"),
                            ("sp|Q5ZKK2|GEMI4_CHICK API_KEY=abcd1234efgh", "abcd1234efgh"),
                            ("OS=Gallus_gallus password=hunter22", "hunter22"), ("x-api-key: 'abcd1234efgh'", "abcd1234efgh")):
            with self.subTest(text=text):
                self.assertNotIn(value, store.redact_text(text))
        self.assertIn("OS=Gallus_gallus", store.redact_text("OS=Gallus_gallus password=hunter22"))

    def test_a_command_line_loses_the_value_after_a_secret_flag_and_any_key_shape(self):
        argv = ["tool.py", "--api-key", "abc", "--token=xyz12345", "--name", "ok", "--key", "k1", "sk-abcdefghijklmnopqrstuvwxyz"]
        redacted = store.redact_argv(argv)
        self.assertEqual(redacted[:3], ["tool.py", "--api-key", store.MASK])
        self.assertNotIn("xyz12345", " ".join(redacted))
        self.assertEqual(redacted[4:6], ["--name", "ok"])
        self.assertEqual(redacted[6:8], ["--key", store.MASK])
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", redacted)

    def test_everything_written_goes_through_the_redaction(self):
        steps = store.StepReader(self.root)
        log = store.EventLog(self.root / "events.jsonl", {"pid": 1, "kind": "x"}, steps)
        log.write("note", text="key sk-abcdefghijklmnopqrstuvwxyz here", nested={"a": ["hf_abcdefghijklmnopqrstuvwxyz012345"]})
        log.close()
        text = (self.root / "events.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("sk-abcdef", text)
        self.assertNotIn("hf_abcdef", text)
        store.atomic_write_json(self.root / "a.json", {"x": "sk-abcdefghijklmnopqrstuvwxyz"})
        self.assertNotIn("sk-abcdef", (self.root / "a.json").read_text(encoding="utf-8"))

    def test_steps_number_themselves_and_the_reader_follows_the_file(self):
        reader = store.StepReader(self.root)
        self.assertEqual(reader.current()["seq"], 0)
        store.write_step(self.root, "first")
        self.assertEqual(reader.current()["label"], "first")
        store.write_step(self.root, "second")
        self.assertEqual((reader.current()["seq"], reader.current()["label"]), (2, "second"))
        self.assertEqual([step["label"] for step in store.read_steps(self.root)], ["first", "second"])

    def test_a_half_written_last_line_of_the_event_log_is_skipped(self):
        path = self.root / "events.jsonl"
        path.write_text('{"ev": "a"}\n{"ev": "b"}\n{"ev": "c', encoding="utf-8")
        self.assertEqual([event["ev"] for event in store.read_events(path)], ["a", "b"])

    def test_request_names_carry_their_target_and_id(self):
        self.assertEqual(store.request_name({"pid": 12}, "abc"), "pid-12--abc.json")
        self.assertEqual(store.request_name({"kind": "viewer"}, "abc"), "kind-viewer--abc.json")
        path = store.write_request(self.root, {"pid": 5}, {"action": "windows"}, "r1")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"action": "windows", "id": "r1"})

    def test_requests_are_ordered_by_when_they_were_written_and_then_by_name(self):
        paths = [self.root / name for name in ("a.json", "b.json", "c.json")]
        for path in paths:
            path.write_text("{}", encoding="utf-8")
        for path, stamp in zip(paths, (30, 10, 20)):
            os.utime(path, ns=(stamp * 10 ** 9, stamp * 10 ** 9))
        self.assertEqual([path.name for path in sorted(paths, key=store.request_order)], ["b.json", "c.json", "a.json"])
        for path in paths:
            os.utime(path, ns=(10 ** 9, 10 ** 9))  # the same instant: the name decides
        self.assertEqual([path.name for path in sorted(reversed(paths), key=store.request_order)], ["a.json", "b.json", "c.json"])
        self.assertEqual(store.request_order(self.root / "gone.json"), (0, "gone.json"))

    def test_the_process_that_is_running_is_alive_and_a_made_up_one_is_not(self):
        self.assertTrue(store.process_alive({"pid": os.getpid(), "created": store.own_start_time()}))
        self.assertFalse(store.process_alive({"pid": 2 ** 22 + 12345}))
        self.assertFalse(store.process_alive({"pid": os.getpid(), "created": 12345.0}))  # a reused pid has another start time

    def test_a_process_knows_when_it_began_and_who_started_it_without_psutil(self):
        import psutil
        started = store.own_start_time()
        if os.name == "nt" or sys.platform.startswith("linux"):
            self.assertIsNotNone(started)
            self.assertAlmostEqual(started, psutil.Process().create_time(), delta=2.0)
        chain = store.lineage()
        self.assertEqual(chain[0], psutil.Process().parent().pid if os.name == "nt" or sys.platform.startswith("linux") else os.getppid())
        # psutil stops at an ancestor whose pid has been reused (a long-lived session, as on a build machine) and the
        # lineage goes on past it, so the two are compared as far as both reach.
        known = [parent.pid for parent in psutil.Process().parents()]
        shared = min(len(chain), len(known))
        self.assertGreaterEqual(shared, 1)
        self.assertEqual(chain[:shared], known[:shared])

    def test_a_replacement_that_collides_with_a_reader_is_tried_again_and_a_failure_leaves_no_temporary_file(self):
        target = self.root / "answer.json"
        real_replace = os.replace
        calls = []

        def busy_twice(source, destination):
            calls.append(source)
            if len(calls) < 3:
                raise PermissionError(5, "Access is denied")
            return real_replace(source, destination)

        with mock.patch.object(store.os, "replace", busy_twice):
            store.atomic_write_json(target, {"a": 1}, attempts=5, pause=0)
        self.assertEqual(store.read_json(target), {"a": 1})
        with mock.patch.object(store.os, "replace", side_effect=PermissionError(5, "Access is denied")):
            with self.assertRaises(PermissionError):
                store.atomic_write_json(target, {"a": 2}, attempts=1)
        self.assertEqual(store.read_json(target), {"a": 1})
        self.assertEqual([path.name for path in self.root.iterdir()], ["answer.json"])


# ---------------------------------------------------------------------------------------------
# Where the probe acts
# ---------------------------------------------------------------------------------------------

class BootDecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "clone"
        make_checkout(self.root)

    def kind(self, relative, argv_tail=(), environ=None):
        script = str(self.root / relative)
        return boot.decide(str(self.root), [script, *argv_tail], environ or {})

    def test_every_entry_point_of_the_checkout_has_a_kind(self):
        expected = {"src/EMAPSSN_Config.py": "config", "src/EMAPSSN_Tools.py": "tools", "src/EMAPSSN_Viewer.py": "viewer",
                    "src/Layout_Cache_Generator.py": "layout", "src/EMAPSSN_MCP_Server.py": "mcp",
                    "src/desktop/Desktop_Launcher.py": "splash", "src/desktop/Desktop_Launcher_Monitor.py": "monitor"}
        for relative, kind in expected.items():
            with self.subTest(script=relative):
                self.assertEqual(self.kind(relative), (kind, None))
        for relative in expected:  # and each of them is a real script of the project
            self.assertTrue((PROJECT_ROOT / relative).is_file(), relative)

    def test_the_launcher_says_which_window_it_starts(self):
        self.assertEqual(self.kind("src/desktop/Desktop_Launcher.py", ["session", "--settings", "x"]), ("splash", "session"))
        self.assertEqual(self.kind("src/desktop/Desktop_Launcher.py", ["viewer"]), ("splash", "viewer"))
        self.assertEqual(self.kind("src/desktop/Desktop_Launcher.py", ["--console"]), ("splash", None))

    def test_the_mcp_servers_headless_runs_of_config_and_tools_are_told_apart_and_its_viewer_is_a_viewer(self):
        launch = ["--headless", "launch-viewer", "--settings", "s.json", "--delete-settings", "--viewer-mode", "hidden"]
        self.assertEqual(self.kind("src/EMAPSSN_Config.py", launch), ("viewer", "headless launch-viewer"))
        self.assertEqual(self.kind("src/EMAPSSN_Config.py", ["--headless", "export", "--kind", "viewer"]), ("config", "headless export"))
        self.assertEqual(self.kind("src/EMAPSSN_Tools.py", ["--headless", "export", "--tool", "x"]), ("tools", "headless export"))
        self.assertEqual(self.kind("src/EMAPSSN_Config.py", ["--headless"]), ("config", "headless"))
        self.assertEqual(self.kind("src/EMAPSSN_Config.py"), ("config", None))

    def test_any_other_script_or_command_is_left_alone(self):
        for argv in (["-c"], ["-m"], [""], [str(self.root / "src" / "Install_Dependencies.py")],
                     [str(self.root / "tests" / "run_sections.py")], [str(self.root / "src" / "tools" / "Align.py")],
                     [str(Path(self.temp.name) / "other" / "EMAPSSN_Config.py")], ["pip"], []):
            with self.subTest(argv=argv):
                self.assertIsNone(boot.decide(str(self.root), argv, {}))

    def test_a_test_run_and_the_kill_switch_keep_the_probe_out(self):
        self.assertIsNone(self.kind("src/EMAPSSN_Config.py", environ={boot.TEST_ENVIRONMENT: "123"}))
        self.assertIsNone(self.kind("src/EMAPSSN_Config.py", environ={boot.KILL_SWITCH: "0"}))
        self.assertEqual(self.kind("src/EMAPSSN_Config.py", environ={boot.KILL_SWITCH: "1"}), ("config", None))

    def test_a_relative_script_path_and_a_different_case_still_match(self):
        with contextlib.chdir(self.root):
            self.assertEqual(boot.entry_kind(str(self.root), os.path.join("src", "EMAPSSN_Tools.py")), "tools")
        self.assertEqual(boot.entry_kind(str(self.root), str(self.root / "SRC" / "emapssn_tools.PY")), "tools")

    def test_the_package_is_registered_without_touching_sys_path_and_the_folder_is_not_a_package(self):
        before = list(sys.path)
        package = boot.load_package(str(PROBE_DIR))
        self.assertEqual(sys.path, before)
        self.assertEqual(package.__path__, [str(PROBE_DIR)])
        # unittest discovery walks into a folder only if it is a package, and imports only test_*.py in it
        self.assertFalse((PROBE_DIR / "__init__.py").exists())
        self.assertEqual(list(PROBE_DIR.glob("test_*.py")), [])


# ---------------------------------------------------------------------------------------------
# Installing, and processes that start with it installed
# ---------------------------------------------------------------------------------------------

class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = make_checkout(Path(self.temp.name) / "clone")
        self.site = self.root / ".venv" / "Lib" / "site-packages"

    def test_install_writes_one_pth_line_and_a_stub_and_uninstall_removes_both(self):
        written = probe_cli.install(self.root)
        self.assertEqual(len(written), 1)
        line = written[0].read_text(encoding="utf-8")
        self.assertEqual(len(line.strip().splitlines()), 1)
        self.assertTrue(line.startswith("import os; os.path.exists("))
        self.assertIn("emapssn_desktop_probe_stub", line)
        self.assertTrue((self.site / probe_cli.STUB_NAME).is_file())
        self.assertEqual(probe_cli.installed(self.root), written)
        probe_cli.uninstall(self.root)
        self.assertEqual(sorted(path.name for path in self.site.iterdir()), [])
        self.assertEqual(probe_cli.installed(self.root), [])

    def test_two_checkouts_sharing_a_venv_keep_each_others_hooks(self):
        other = make_checkout(Path(self.temp.name) / "other")
        shutil.rmtree(other / ".venv")
        os.symlink(self.root / ".venv", other / ".venv", target_is_directory=True) if os.name != "nt" else None
        if os.name == "nt":
            self.skipTest("a venv shared through a symlink needs the privilege to make one")
        probe_cli.install(self.root)
        probe_cli.install(other)
        self.assertEqual(len(list(self.site.glob("emapssn_desktop_probe-*.pth"))), 2)
        probe_cli.uninstall(self.root)
        self.assertEqual(len(list(self.site.glob("emapssn_desktop_probe-*.pth"))), 1)
        self.assertTrue((self.site / probe_cli.STUB_NAME).is_file())
        probe_cli.uninstall(other)
        self.assertEqual(list(self.site.iterdir()), [])

    def test_a_posix_layout_is_found_too(self):
        shutil.rmtree(self.root / ".venv")
        posix = self.root / ".venv" / "lib" / "python3.13" / "site-packages"
        posix.mkdir(parents=True)
        self.assertEqual(probe_cli.site_packages(self.root), [posix])
        probe_cli.install(self.root)
        self.assertEqual(len(list(posix.glob("emapssn_desktop_probe-*.pth"))), 1)

    def test_a_checkout_without_a_venv_is_refused(self):
        shutil.rmtree(self.root / ".venv")
        with self.assertRaises(SystemExit):
            probe_cli.install(self.root)

    # --- running Python with the hook in site-packages

    def start(self, script_relative, body, *, arguments=(), environment=None, gate=True, wait=True):
        """Run ``body`` as the script ``script_relative`` of the checkout, with the venv's site-packages added as site does."""
        probe_cli.install(self.root)
        if gate:
            store.write_gate(self.root, self.root / "out", settle_ms=50)
        script = self.root / script_relative
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(f"import site\nsite.addsitedir({str(self.site)!r})\n" + textwrap.dedent(body), encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-u", str(script), *arguments], cwd=self.root, env=environment or clean_environment(),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(self.stop, process)
        if wait:
            output, _ = process.communicate(timeout=60)
            return process, output
        return process, ""

    @staticmethod
    def stop(process):
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdout:
            process.stdout.close()

    def test_a_venv_with_the_probe_installed_and_the_gate_off_loads_nothing_and_writes_nothing(self):
        _process, output = self.start("src/EMAPSSN_Tools.py", f"""
            import sys
            ours = [name for name, module in sys.modules.items() if name != "__main__" and {str(self.root)!r} in str(getattr(module, "__file__", ""))]
            print("MODULES", ours)
            """, gate=False)
        self.assertIn("MODULES []", output)
        self.assertFalse((self.root / "temp").exists())
        self.assertFalse((self.root / "out").exists())

    def test_with_the_gate_on_a_script_that_is_no_entry_point_is_left_alone(self):
        self.start("src/tools/Something.py", "pass\n")
        self.assertEqual(self.recorded_folders(), [])

    def test_a_process_of_a_test_run_stays_inert_even_when_it_runs_an_entry_point(self):
        self.start("src/EMAPSSN_Tools.py", "pass\n", environment=clean_environment(**{boot.TEST_ENVIRONMENT: "4242"}))
        self.assertEqual(self.recorded_folders(), [])

    def recorded_folders(self):
        procs = self.root / "out" / "procs"
        return sorted(path.name for path in procs.iterdir()) if procs.exists() else []

    def test_the_hook_run_twice_by_a_python_that_runs_pth_lines_twice_starts_one_probe(self):
        _process, output = self.start("src/desktop/Desktop_Launcher.py", f"""
            import site
            site.addsitedir({str(self.site)!r})
            """)
        folders = self.recorded_folders()
        self.assertEqual(len(folders), 1, folders)
        self.assertTrue(folders[0].endswith("-splash"))

    def test_two_checkouts_that_share_one_venv_and_so_one_stub_each_get_their_own_probe(self):
        other = make_checkout(Path(self.temp.name) / "other")
        store.write_gate(other, other / "out")
        store.write_gate(self.root, self.root / "out")
        stub_folder = self.root / "shared"
        stub_folder.mkdir()
        shutil.copyfile(PROBE_DIR / "stub.py", stub_folder / probe_cli.STUB_NAME)
        script = self.root / "src" / "EMAPSSN_Tools.py"
        script.write_text(textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(stub_folder)!r})
            import emapssn_desktop_probe_stub as stub
            stub.load({str(other)!r})   # the other checkout's .pth runs first: this script is not its entry point
            stub.load({str(self.root)!r})
            stub.load({str(self.root)!r})
            """), encoding="utf-8")
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=self.root, env=clean_environment())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.recorded_folders()), 1)
        self.assertFalse((other / "out" / "procs").exists() and any((other / "out" / "procs").iterdir()))

    def test_an_entry_point_records_its_start_and_exit_with_the_command_line_redacted(self):
        process, _output = self.start("src/EMAPSSN_Tools.py", "pass\n",
                                      arguments=["--api-key", "very-secret-value", "sk-abcdefghijklmnopqrstuvwxyz"])
        self.assertEqual(process.returncode, 0)
        (folder,) = list((self.root / "out" / "procs").iterdir())
        record = store.read_json(folder / "process.json")
        self.assertEqual(record["kind"], "tools")
        self.assertEqual(record["pid"], int(folder.name.split("-")[0]))
        events = store.read_events(folder / "events.jsonl")
        self.assertEqual([event["ev"] for event in events][:1], ["process_start"])
        self.assertEqual(events[-1]["ev"], "exit")
        self.assertGreaterEqual(events[-1]["mono_ms"], events[0]["mono_ms"])
        text = (folder / "events.jsonl").read_text(encoding="utf-8") + (folder / "process.json").read_text(encoding="utf-8")
        self.assertNotIn("very-secret-value", text)
        self.assertNotIn("sk-abcdef", text)
        self.assertIn(store.MASK, text)

    def test_os_exit_still_leaves_an_exit_record_and_a_forked_worker_leaves_none(self):
        self.start("src/EMAPSSN_Tools.py", """
            import os
            os._exit(3)
            """)
        (folder,) = list((self.root / "out" / "procs").iterdir())
        events = store.read_events(folder / "events.jsonl")
        self.assertEqual([event["ev"] for event in events[-2:]], ["exit_code", "exit"])
        self.assertEqual(events[-2]["code"], 3)

    @unittest.skipUnless(hasattr(os, "fork"), "fork exists on POSIX only")
    def test_a_forked_worker_does_not_write_into_its_parents_record(self):
        self.start("src/EMAPSSN_Tools.py", """
            import os
            pid = os.fork()
            if pid == 0:
                os._exit(0)
            os.waitpid(pid, 0)
            """)
        (folder,) = list((self.root / "out" / "procs").iterdir())
        exits = [event for event in store.read_events(folder / "events.jsonl") if event["ev"] == "exit"]
        self.assertEqual(len(exits), 1)

    def test_a_failure_to_load_the_probe_never_reaches_the_program(self):
        probe_cli.install(self.root)
        store.write_gate(self.root, self.root / "out")
        (self.root / "tests" / "desktop_probe" / "boot.py").write_text("raise RuntimeError('broken probe')\n", encoding="utf-8")
        script = self.root / "src" / "EMAPSSN_Tools.py"
        script.write_text(f"import site\nsite.addsitedir({str(self.site)!r})\nprint('PROGRAM RAN')\n", encoding="utf-8")
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=clean_environment(), cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PROGRAM RAN", result.stdout)
        self.assertEqual(result.stderr.strip(), "")
        self.assertIn("broken probe", (store.home_dir(self.root) / "boot-errors.log").read_text(encoding="utf-8"))

    def test_qt_hand_over_in_a_process_that_makes_its_application_late(self):
        process, output = self.start("src/EMAPSSN_Tools.py", """
            import time
            time.sleep(0.3)                      # the Viewer imports PyTorch before it makes its QApplication
            from PySide6 import QtCore, QtWidgets
            app = QtWidgets.QApplication([])
            window = QtWidgets.QWidget()
            window.setWindowTitle("Late window")
            window.show()
            QtCore.QTimer.singleShot(1200, app.quit)
            app.exec()
            print("THREADS", [thread.name for thread in __import__("threading").enumerate()])
            """)
        self.assertEqual(process.returncode, 0, output)
        self.assertNotIn("desktop-probe-wait", output)  # the thread that waited for the application has ended
        (folder,) = list((self.root / "out" / "procs").iterdir())
        events = store.read_events(folder / "events.jsonl")
        names = [event["ev"] for event in events]
        self.assertIn("app_ready", names)
        shown = next(event for event in events if event["ev"] == "window_shown")
        self.assertEqual(shown["window"]["title"], "Late window")
        self.assertIn("capture", names)
        self.assertTrue(any((self.root / "out" / "captures").glob("tools-*-Late-window.png")))
        self.assertEqual(names[-1], "exit")



# ---------------------------------------------------------------------------------------------
# The in-process part, on this process's QApplication
# ---------------------------------------------------------------------------------------------

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402


class FileAgent:
    """What a test agent does, with no Qt: a request file in, a response file out (the way probe.py do works)."""

    def __init__(self, out, pid):
        self.out = out
        self.pid = pid

    def ask(self, request, timeout=15.0):
        request_id = uuid.uuid4().hex[:8]
        store.write_request(self.out, {"pid": self.pid}, request, request_id)
        path = store.response_path(self.out, request_id)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = store.read_json(path)
            if response is not None:
                return response
            time.sleep(0.01)
        raise AssertionError(f"no answer to {request}")


class ProbeCase(unittest.TestCase):
    """A Probe running in this process, writing under a temporary folder; windows are made with make_window."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from emapssn_desktop_probe import runtime
        cls.runtime = runtime

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        store.write_gate(self.root, self.root / "out", settle_ms=20)
        self.recorder = recorder.Recorder(self.root, "tools", store.read_gate(self.root), ["tools.py"])
        self.recorder.prepare()
        self.out = self.recorder.out
        self.probe = self.runtime.Probe(self.recorder, self.app)
        self.probe.start()
        self.made = []
        self.addCleanup(self.shut_down)

    def shut_down(self):
        for window in self.made:
            with contextlib.suppress(RuntimeError):
                window.close()
                window.deleteLater()
        self.app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.probe.shutdown()
        self.recorder.finish("test")

    # --- helpers
    def make_window(self, title, build=None, *, show=True, size=(320, 240)):
        window = QtWidgets.QWidget()
        window.setWindowTitle(title)
        window.resize(*size)
        layout = QtWidgets.QVBoxLayout(window)
        if build is not None:
            build(window, layout)
        if show:
            window.show()
        self.made.append(window)
        return window

    def pump(self, milliseconds=80):
        end = time.monotonic() + milliseconds / 1000
        while time.monotonic() < end:
            self.app.processEvents()
            self.probe._tick()
            time.sleep(0.004)

    def events(self, name=None):
        events = store.read_events(self.recorder.folder / "events.jsonl")
        return [event for event in events if name is None or event["ev"] == name]

    def ask(self, request, *, timeout=15.0, **options):
        """Write a request file and run the event loop (and the probe's tick) until its answer file appears."""
        request = {"settle_ms": 0, **request, **options}
        request_id = uuid.uuid4().hex[:8]
        store.write_request(self.out, {"pid": os.getpid()}, request, request_id)
        path = store.response_path(self.out, request_id)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.probe._tick()
            response = store.read_json(path)
            if response is not None:
                return response
            time.sleep(0.004)
        errors = self.recorder.folder / "errors.log"
        raise AssertionError(
            f"no answer to {request}; probe disabled: {self.recorder.disabled}; "
            f"errors.log: {errors.read_text(encoding='utf-8')[-1500:] if errors.exists() else None}; "
            f"windows: {[(type(w).__name__, w.windowTitle()) for w in self.app.topLevelWidgets() if w.isVisible()]}")

    def ok(self, request, **options):
        response = self.ask(request, **options)
        self.assertTrue(response["ok"], response)
        return response["result"]


class WindowEventTests(ProbeCase):
    def test_a_window_is_recorded_when_shown_painted_hidden_closed_and_gone(self):
        window = self.make_window("Events window")
        self.pump()
        shown = next(event for event in self.events("window_shown") if event["window"]["title"] == "Events window")
        description = shown["window"]
        self.assertEqual(description["class"], "QWidget")
        self.assertEqual(len(description["geometry"]), 4)
        self.assertEqual(description["geometry"][2:], [320, 240])
        self.assertEqual(len(description["min_size"]), 2)
        self.assertGreater(description["dpr"], 0)
        self.assertIn("screen", description)
        self.assertIn("active", description)
        self.assertTrue(shown["first"])
        for name in ("wall_ms", "mono_ms", "pid", "kind", "serial"):
            self.assertIn(name, shown)
        if window.windowHandle().isExposed():
            self.assertTrue(any(event["window"]["title"] == "Events window" for event in self.events("window_first_painted")))
        window.hide()
        self.pump()
        self.assertTrue(any(event["window"]["title"] == "Events window" for event in self.events("window_hidden")))
        window.show()
        self.pump()
        window.close()
        self.pump()
        self.assertTrue(any(event["window"]["title"] == "Events window" for event in self.events("window_close_requested")))
        self.assertTrue(any(event["window"]["title"] == "Events window" for event in self.events("window_closed")))
        window.deleteLater()
        self.made.remove(window)
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.pump()
        self.assertTrue(any(event["window"]["title"] == "Events window" for event in self.events("window_destroyed")))

    def test_geometry_and_title_changes_are_recorded(self):
        window = self.make_window("Changes window")
        self.pump()
        window.resize(400, 300)
        window.setWindowTitle("Changes window renamed")
        self.pump()
        changes = [event["changed"] for event in self.events("window_changed")]
        self.assertTrue(any("geometry" in change for change in changes), changes)
        self.assertTrue(any("title" in change for change in changes), changes)

    def test_the_focus_widget_is_recorded_with_whether_it_draws_focus(self):
        def build(window, layout):
            layout.addWidget(QtWidgets.QLineEdit())
        window = self.make_window("Focus window", build)
        window.activateWindow()
        window.findChild(QtWidgets.QLineEdit).setFocus()
        self.pump()
        focus = [event["focus"] for event in self.events("focus_changed") if event["focus"]]
        self.assertTrue(any(item["class"] == "QLineEdit" and "draws_focus" in item for item in focus), focus)

    def test_a_message_box_or_dialog_is_recorded_with_its_text_and_buttons(self):
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Warning, "Careful", "The cache is old.",
                                    QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setInformativeText("Rebuild it?")
        self.made.append(box)
        box.show()
        self.pump()
        dialog = next(event["dialog"] for event in self.events("dialog_shown") if event["dialog"]["title"] == "Careful")
        self.assertEqual(dialog["text"], "The cache is old.")
        self.assertEqual(dialog["informative_text"], "Rebuild it?")
        self.assertEqual(sorted(dialog["buttons"]), ["Cancel", "OK"])
        self.assertEqual(dialog["icon"], "Warning")
        plain = QtWidgets.QDialog()
        plain.setWindowTitle("Plain dialog")
        QtWidgets.QVBoxLayout(plain).addWidget(QtWidgets.QPushButton("Go on"))
        self.made.append(plain)
        plain.show()
        self.pump()
        info = next(event["dialog"] for event in self.events("dialog_shown") if event["dialog"]["title"] == "Plain dialog")
        self.assertEqual(info["buttons"], ["Go on"])

    def test_the_event_filter_never_swallows_an_event(self):
        seen = []

        class Watched(QtWidgets.QWidget):
            def showEvent(self, event):
                seen.append("show")
                super().showEvent(event)

            def closeEvent(self, event):
                seen.append("close")
                super().closeEvent(event)

        window = Watched()
        window.setWindowTitle("Watched window")
        self.made.append(window)
        window.show()
        self.pump()
        window.close()
        self.pump()
        self.assertEqual(seen, ["show", "close"])
        for kind in (QtCore.QEvent.Type.Show, QtCore.QEvent.Type.Hide, QtCore.QEvent.Type.Close, QtCore.QEvent.Type.Expose):
            self.assertIs(self.probe.windows._watcher.eventFilter(window, QtCore.QEvent(kind)), False)


class CaptureTests(ProbeCase):
    def test_a_new_window_is_captured_after_the_settle_time_with_a_name_that_says_what_and_when(self):
        store.write_step(self.out, "after the click")
        self.make_window("Capture window")
        self.pump(150)
        files = list((self.out / "captures").glob("tools-*-Capture-window.png"))
        self.assertEqual(len(files), 1, [path.name for path in (self.out / "captures").iterdir()])
        self.assertRegex(files[0].name, rf"^tools-{os.getpid()}-0001-after-the-click-Capture-window\.png$")
        image = QtGui.QImage(str(files[0]))
        self.assertFalse(image.isNull())
        capture = next(event for event in self.events("capture") if event["title"] == "Capture window")
        self.assertEqual(capture["reason"], "auto")
        self.assertEqual(capture["step"], "after the click")
        self.assertEqual(capture["width"], image.width())
        self.assertGreater(capture["dpr"], 0)
        self.assertEqual(capture["logical"], [320, 240])

    def test_the_picture_has_the_screens_device_pixel_ratio(self):
        self.make_window("Scaled window")
        self.pump()
        result = self.ok({"action": "capture", "window": "Scaled window"})
        dpr = result["dpr"]
        self.assertAlmostEqual(result["width"], 320 * dpr, delta=2)
        self.assertEqual(result["logical"], [320, 240])
        self.assertEqual(QtGui.QImage(result["path"]).width(), result["width"])

    def test_a_capture_can_be_of_one_widget_or_of_every_window(self):
        def build(window, layout):
            button = QtWidgets.QPushButton("Only me")
            button.setObjectName("only")
            layout.addWidget(button)
        self.make_window("Widget window", build)
        self.make_window("Second window")
        self.pump()
        one = self.ok({"action": "capture", "window": "Widget window", "widget": "only"})
        self.assertLess(one["width"], 320 * one["dpr"])
        self.assertIn("only", one["title"].lower())
        everything = self.ok({"action": "capture", "all": True})
        titles = [shot["window"] for shot in everything["captures"]]
        self.assertIn("Widget window", titles)
        self.assertIn("Second window", titles)

    def test_auto_capture_can_be_switched_off_and_a_window_that_looks_like_it_holds_a_key_is_not_captured(self):
        self.probe.auto_capture = False
        self.make_window("Quiet window")
        self.pump(120)
        self.assertEqual(list((self.out / "captures").glob("*Quiet*")), [])
        self.probe.auto_capture = True
        self.make_window("Biohub API token")
        self.pump(120)
        self.assertEqual(list((self.out / "captures").glob("*Biohub*")), [])
        self.assertTrue(self.events("capture_skipped"))
        refused = self.ask({"action": "capture", "window": "Biohub API token"})
        self.assertFalse(refused["ok"])
        self.assertIn("force", refused["error"])
        self.assertTrue(self.ok({"action": "capture", "window": "Biohub API token", "force": True})["file"])

    def test_a_window_that_is_never_mapped_is_captured_too(self):
        window = self.make_window("Hidden window", show=False)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()
        self.pump(150)
        shown = next(event for event in self.events("window_shown") if event["window"]["title"] == "Hidden window")
        self.assertTrue(shown["window"]["offscreen_only"])
        self.assertTrue(list((self.out / "captures").glob("*Hidden-window.png")))

    def test_a_flat_picture_and_the_widgets_qt_does_not_paint_are_reported(self):
        from emapssn_desktop_probe import capture
        image = QtGui.QImage(40, 40, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.QColor("#000000"))
        self.assertEqual(capture.flat_colour(image, image.rect()), "#000000")
        for x in range(20, 40):
            for y in range(40):
                image.setPixelColor(x, y, QtGui.QColor("#ffffff"))
        self.assertIsNone(capture.flat_colour(image, QtCore.QRect(0, 0, 40, 40)))
        self.assertEqual(capture.flat_colour(image, QtCore.QRect(0, 0, 10, 10)), "#000000")

        class QOpenGLWidget(QtWidgets.QWidget):  # stands in for the VisPy canvas: a class of that name
            pass

        def build(window, layout):
            canvas = QOpenGLWidget()
            canvas.setObjectName("canvas")
            canvas.setMinimumSize(80, 60)
            layout.addWidget(canvas)
        self.make_window("Canvas window", build)
        self.pump()
        result = self.ok({"action": "capture", "window": "Canvas window"})
        (part,) = result["drawn_elsewhere"]
        self.assertEqual((part["class"], part["objectName"], part["kind"]), ("QOpenGLWidget", "canvas", "QOpenGLWidget"))


class TargetTests(ProbeCase):
    def test_a_window_is_found_by_title_class_index_and_the_newest_wins_without_a_name(self):
        self.make_window("Alpha window")
        self.pump()  # a scan numbers the windows it finds in turn; two found in one scan have no order
        self.make_window("Beta window")
        self.pump()
        windows = self.ok({"action": "windows"})["windows"]
        titles = [entry["title"] for entry in windows]
        self.assertLess(titles.index("Alpha window"), titles.index("Beta window"))
        index = titles.index("Beta window")
        self.assertEqual(self.ok({"action": "tree", "window": index})["window"], "Beta window")
        self.assertEqual(self.ok({"action": "tree", "window": {"title": "^alpha"}})["window"], "Alpha window")
        missing = self.ask({"action": "tree", "window": "No such window"})
        self.assertFalse(missing["ok"])
        self.assertIn("Alpha window", missing["detail"]["windows"])

    def test_a_widget_is_found_by_object_name_accessible_name_text_tooltip_and_path(self):
        clicks = []

        def build(window, layout):
            button = QtWidgets.QPushButton("Run &it")
            button.setObjectName("run")
            button.setToolTip("Runs the thing")
            button.setAccessibleName("runner")
            button.clicked.connect(lambda: clicks.append(1))
            layout.addWidget(button)
        self.make_window("Find window", build)
        self.pump()
        for widget in ("run", "Run it", "Runs the thing", "runner", {"accessibleName": "runner"}, {"text": "run it"},
                       {"class": "QPushButton"}, {"toolTip": "thing"}, {"path": "0"}):
            with self.subTest(widget=widget):
                self.ok({"action": "click", "window": "Find window", "widget": widget})
        self.assertEqual(len(clicks), 9)
        tree = self.ok({"action": "tree", "window": "Find window"})["tree"]
        button = tree["children"][0]
        self.assertEqual((button["class"], button["objectName"], button["accessibleName"]), ("QPushButton", "run", "runner"))
        self.assertEqual((button["text"], button["toolTip"], button["path"]), ("Run &it", "Runs the thing", "0"))
        self.assertTrue(button["visible"] and button["enabled"])
        self.assertEqual(len(button["geometry"]), 4)

    def test_a_window_name_that_matches_several_windows_is_answered_with_a_note_saying_which_was_used(self):
        self.make_window("Report window")
        self.pump()
        self.make_window("Report window - details")
        self.pump()
        answer = self.ask({"action": "tree", "window": "^Report window"})
        self.assertTrue(answer["ok"], answer)
        self.assertIn("2 windows matched", answer["window_note"])
        self.assertIn("'Report window - details'", answer["window_note"])  # the newest of them
        exact = self.ask({"action": "tree", "window": "^Report window$"})
        self.assertNotIn("window_note", exact)
        self.assertEqual(exact["result"]["window"], "Report window")
        closed = self.ask({"action": "close", "window": "Report window"})  # a deferred action carries the note too
        self.assertIn("window_note", closed)

    def test_nothing_found_and_too_many_found_are_errors_that_list_what_is_there(self):
        def build(window, layout):
            for _ in range(2):
                layout.addWidget(QtWidgets.QPushButton("Same"))
            layout.addWidget(QtWidgets.QPushButton("Different"))
        self.make_window("Many window", build)
        self.pump()
        none = self.ask({"action": "click", "window": "Many window", "widget": "Nothing like it"})
        self.assertFalse(none["ok"])
        self.assertTrue(any(row["text"] == "Different" for row in none["detail"]["visible_candidates"]))
        many = self.ask({"action": "click", "window": "Many window", "widget": "Same"})
        self.assertFalse(many["ok"])
        self.assertEqual(len(many["detail"]["matches"]), 2)
        self.assertTrue(self.ok({"action": "click", "window": "Many window", "widget": {"text": "Same", "index": 1}}))

    def test_a_tab_a_list_row_and_a_menu_entry_are_targets_too(self):
        chosen = []
        holder = {}

        def build(window, layout):
            tabs = QtWidgets.QTabWidget()
            tabs.addTab(QtWidgets.QLabel("one"), "General")
            tabs.addTab(QtWidgets.QLabel("two"), "Advanced")
            holder["tabs"] = tabs
            items = QtWidgets.QListWidget()
            items.addItems(["alpha", "beta", "gamma"])
            holder["items"] = items
            layout.addWidget(tabs)
            layout.addWidget(items)
        self.make_window("Parts window", build)
        self.pump()
        self.ok({"action": "click", "window": "Parts window", "widget": {"tab": "Advanced"}})
        self.assertEqual(holder["tabs"].currentIndex(), 1)
        self.ok({"action": "click", "window": "Parts window", "widget": {"item": "gamma"}})
        self.assertEqual(holder["items"].currentRow(), 2)
        menu = QtWidgets.QMenu()
        menu.addAction("Copy", lambda: chosen.append("copy"))
        menu.addAction("Paste", lambda: chosen.append("paste"))
        self.made.append(menu)
        menu.popup(QtCore.QPoint(20, 20))
        self.pump()
        self.ok({"action": "click", "window": {"class": "QMenu"}, "widget": {"action": "Paste"}})
        self.assertEqual(chosen, ["paste"])


class ActionTests(ProbeCase):
    def test_a_click_is_a_mouse_event_through_the_window_and_says_what_was_under_the_pointer(self):
        clicks = []

        def build(window, layout):
            button = QtWidgets.QPushButton("Go")
            button.clicked.connect(lambda: clicks.append(1))
            layout.addWidget(button)
        self.make_window("Click window", build)
        self.pump()
        result = self.ok({"action": "click", "window": "Click window", "widget": "Go"})
        self.assertEqual(len(clicks), 1)
        self.assertTrue(result["target_under_pointer"])
        self.assertEqual(result["widget_under_pointer"]["class"], "QPushButton")
        self.assertEqual(result["window"], "Click window")
        self.assertIn("windows_after", result)
        self.assertEqual(self.ok({"action": "click", "window": "Click window", "widget": "Go", "double": True})["window"], "Click window")

    def test_a_click_on_a_disabled_button_says_so(self):
        def build(window, layout):
            button = QtWidgets.QPushButton("Off")
            button.setEnabled(False)
            layout.addWidget(button)
        self.make_window("Disabled window", build)
        self.pump()
        self.assertIn("disabled", self.ok({"action": "click", "window": "Disabled window", "widget": "Off"})["warning"])

    def test_a_click_gives_the_keyboard_focus_as_it_does_for_a_user(self):
        def build(window, layout):
            layout.addWidget(QtWidgets.QLineEdit())
            second = QtWidgets.QLineEdit()
            second.setObjectName("second")
            layout.addWidget(second)
        window = self.make_window("Focus click window", build)
        self.pump()
        self.ok({"action": "click", "window": "Focus click window", "widget": "second"})
        self.assertEqual(window.focusWidget().objectName(), "second")

    def test_a_link_in_a_label_is_clicked_where_the_link_is(self):
        links = []

        def build(window, layout):
            label = QtWidgets.QLabel('Read <a href="https://example.org/one">the first</a> and <a href="https://example.org/two">the second</a>.')
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
            label.linkActivated.connect(links.append)
            layout.addWidget(label)
        self.make_window("Link window", build)
        self.pump()
        self.ok({"action": "click", "window": "Link window", "widget": {"class": "QLabel"}, "link": "the second"})
        self.ok({"action": "click", "window": "Link window", "widget": {"class": "QLabel"}, "link": "https://example.org/one"})
        self.assertEqual(links, ["https://example.org/two", "https://example.org/one"])
        missing = self.ask({"action": "click", "window": "Link window", "widget": {"class": "QLabel"}, "link": "no such link"})
        self.assertFalse(missing["ok"])

    def test_typing_goes_to_the_named_field_clicks_it_first_and_handles_any_character(self):
        returned = []

        def build(window, layout):
            edit = QtWidgets.QLineEdit()
            edit.setObjectName("name")
            edit.returnPressed.connect(lambda: returned.append(1))
            layout.addWidget(edit)
            layout.addWidget(QtWidgets.QLineEdit())
        window = self.make_window("Type window", build)
        self.pump()
        edit = window.findChild(QtWidgets.QLineEdit, "name")
        self.ok({"action": "type", "window": "Type window", "widget": "name", "text": "Hello é中 x"})
        self.assertEqual(edit.text(), "Hello é中 x")
        self.assertIs(window.focusWidget(), edit)
        self.ok({"action": "type", "window": "Type window", "widget": "name", "text": "New", "clear": True})
        self.assertEqual(edit.text(), "New")
        self.ok({"action": "type", "text": "er"})  # no widget: the one with the focus
        self.assertEqual(edit.text(), "Newer")
        self.ok({"action": "key", "keys": "Return"})
        self.assertEqual(returned, [1])
        self.ok({"action": "key", "keys": ["Ctrl+A", "Delete"], "window": "Type window"})
        self.assertEqual(edit.text(), "")
        bad = self.ask({"action": "key", "keys": "NotAKey"})
        self.assertFalse(bad["ok"])

    def test_a_key_sequence_reaches_a_shortcut(self):
        fired = []

        def build(window, layout):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+K"), window)
            shortcut.activated.connect(lambda: fired.append(1))
            layout.addWidget(QtWidgets.QLineEdit())
        window = self.make_window("Shortcut window", build)
        window.activateWindow()
        self.pump()
        self.ok({"action": "key", "keys": "Ctrl+K", "window": "Shortcut window"})
        self.assertEqual(fired, [1])

    def test_set_value_for_a_combo_spin_boxes_check_boxes_a_toggle_and_a_line_edit(self):
        seen = []
        holder = {}

        def build(window, layout):
            combo = QtWidgets.QComboBox()
            combo.setObjectName("combo")
            combo.addItems(["one", "two", "three"])
            combo.activated.connect(lambda index: seen.append(("activated", index)))
            spin = QtWidgets.QSpinBox()
            spin.setObjectName("spin")
            spin.setMaximum(100)
            decimals = QtWidgets.QDoubleSpinBox()
            decimals.setObjectName("decimals")
            check = QtWidgets.QCheckBox("Keep")
            toggle = QtWidgets.QPushButton("Switch")
            toggle.setCheckable(True)
            toggle.toggled.connect(lambda state: seen.append(("toggled", state)))
            edit = QtWidgets.QLineEdit()
            edit.setObjectName("edit")
            edit.textEdited.connect(lambda text: seen.append(("edited", text)))
            slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            slider.setObjectName("slider")
            for widget in (combo, spin, decimals, check, toggle, edit, slider):
                layout.addWidget(widget)
            holder.update(combo=combo, spin=spin, decimals=decimals, check=check, toggle=toggle, edit=edit, slider=slider)
        self.make_window("Values window", build)
        self.pump()

        def value(widget, wanted):
            return self.ok({"action": "set_value", "window": "Values window", "widget": widget, "value": wanted})
        self.assertEqual(value("combo", "three")["value"], "three")
        self.assertEqual(holder["combo"].currentIndex(), 2)
        self.assertEqual(value("combo", 0)["value"], "one")
        self.assertIn(("activated", 2), seen)
        self.assertEqual(value("spin", 42)["value"], 42)
        self.assertEqual(value("decimals", 1.5)["value"], 1.5)
        value("Keep", True)
        self.assertTrue(holder["check"].isChecked())
        value("Switch", True)
        value("Switch", True)  # already on: nothing happens
        self.assertEqual([item for item in seen if item[0] == "toggled"], [("toggled", True)])
        value("edit", "typed")
        self.assertEqual(holder["edit"].text(), "typed")
        self.assertIn(("edited", "typed"), seen)
        value("slider", 7)
        self.assertEqual(holder["slider"].value(), 7)
        no_entry = self.ask({"action": "set_value", "window": "Values window", "widget": "combo", "value": "nine"})
        self.assertFalse(no_entry["ok"])
        self.assertEqual(no_entry["detail"]["items"], ["one", "two", "three"])

    def test_close_acts_like_the_title_bar_and_says_when_the_window_refused(self):
        class Stubborn(QtWidgets.QWidget):
            refuse = True

            def closeEvent(self, event):
                event.ignore() if self.refuse else event.accept()

        stubborn = Stubborn()
        stubborn.setWindowTitle("Stubborn window")
        self.made.append(stubborn)
        stubborn.show()
        plain = self.make_window("Plain window")
        self.pump()
        self.assertTrue(self.ok({"action": "close", "window": "Plain window"})["accepted"])
        self.assertFalse(plain.isVisible())
        self.assertFalse(self.ok({"action": "close", "window": "Stubborn window"})["accepted"])
        self.assertTrue(stubborn.isVisible())
        stubborn.refuse = False

    def test_resize_to_a_size_and_to_the_minimum_the_layout_allows(self):
        def build(window, layout):
            layout.addWidget(QtWidgets.QLabel("A fairly long label that sets the minimum width"))
            layout.addWidget(QtWidgets.QPushButton("Button"))
        window = self.make_window("Resize window", build, size=(600, 400))
        self.pump()
        result = self.ok({"action": "resize", "window": "Resize window", "width": 700, "height": 350})
        self.assertEqual(result["size"], [700, 350])
        smallest = self.ok({"action": "resize", "window": "Resize window", "to_minimum": True})
        self.assertEqual(smallest["size"], smallest["min_size_hint"])
        self.assertLess(smallest["size"][0], 700)
        self.assertGreater(smallest["size"][0], 100)
        self.assertEqual([window.width(), window.height()], smallest["size"])
        self.assertFalse(self.ask({"action": "resize", "window": "Resize window"})["ok"])

    def test_the_clipboard_is_read_and_can_be_set(self):
        self.assertEqual(self.ok({"action": "clipboard", "set": "copied text"})["text"], "copied text")
        self.assertEqual(self.ok({"action": "clipboard"})["text"], "copied text")
        QtGui.QGuiApplication.clipboard().setText("from the program")
        self.assertEqual(self.ok({"action": "clipboard"})["text"], "from the program")

    def test_eval_reads_state_and_reports_an_error_as_an_answer(self):
        self.make_window("Eval window")
        self.pump()
        self.assertTrue(self.ok({"action": "eval", "expr": "len(windows) >= 1"})["value"])
        self.assertEqual(self.ok({"action": "eval", "expr": "widget.windowTitle()", "window": "Eval window",
                                  "widget": {"class": "QWidget"}})["value"], "Eval window")
        self.assertEqual(self.ok({"action": "eval", "mode": "exec", "expr": "result = [1, 2, 3][1]"})["value"], 2)
        failed = self.ask({"action": "eval", "expr": "1 / 0"})
        self.assertFalse(failed["ok"])
        self.assertIn("ZeroDivisionError", failed["error"])

    def test_a_viewer_command_runs_through_the_viewers_command_line_and_returns_its_status(self):
        from types import SimpleNamespace

        class FakeViewer:
            def __init__(self):
                self.commands = []
                self.console_text = SimpleNamespace(text="")

            def process_command(self, text):
                self.commands.append(text)
                self.console_text.text = f"ran: {text}"

        viewer = FakeViewer()
        main = sys.modules["__main__"]
        with mock.patch.object(main, "viewer", viewer, create=True):
            result = self.ok({"action": "viewer_command", "command": "help color"})
        self.assertEqual(viewer.commands, ["help color"])
        self.assertEqual(result["status_text"], "ran: help color")
        self.assertIn("outcome", result)
        self.assertIn("printed", result)
        refused = self.ask({"action": "viewer_command", "command": "help"})
        self.assertFalse(refused["ok"])
        self.assertIn("no Viewer", refused["error"])

    def test_the_viewers_own_outcome_record_is_used_when_the_engine_is_loaded(self):
        import Command_Engine

        class FakeViewer:
            console_text = type("Text", (), {"text": ""})()

            def process_command(self, text):
                Command_Engine._record_outcome("failed")

        main = sys.modules["__main__"]
        with mock.patch.object(main, "viewer", FakeViewer(), create=True):
            result = self.ok({"action": "viewer_command", "command": "blahblah"})
        self.assertEqual(result["outcome"], "failed")
        self.assertFalse(result["queued_or_pending"])

    def test_requests_for_another_process_are_left_alone_and_a_kind_request_is_answered(self):
        other = store.write_request(self.out, {"pid": os.getpid() + 1}, {"action": "windows"}, "other-pid")
        elsewhere = store.write_request(self.out, {"kind": "viewer"}, {"action": "windows"}, "other-kind")
        store.write_request(self.out, {"kind": "tools"}, {"action": "ping"}, "mine")
        self.pump(200)
        self.assertTrue(other.exists() and elsewhere.exists())
        answer = store.read_json(store.response_path(self.out, "mine"))
        self.assertTrue(answer["ok"])
        self.assertEqual((answer["pid"], answer["kind"]), (os.getpid(), "tools"))
        self.assertFalse(store.response_path(self.out, "other-pid").exists())

    def test_a_request_that_is_still_being_written_is_left_until_it_parses(self):
        path = self.out / "requests" / store.request_name({"pid": os.getpid()}, "slow")
        path.write_text('{"action": "pi', encoding="utf-8")
        self.pump(200)
        self.assertTrue(path.exists())
        self.assertFalse(store.response_path(self.out, "slow").exists())
        path.write_text('{"action": "ping"}', encoding="utf-8")
        deadline = time.monotonic() + 5
        while not store.response_path(self.out, "slow").exists() and time.monotonic() < deadline:
            self.pump(30)
        self.assertTrue(store.read_json(store.response_path(self.out, "slow"))["ok"])

    def write_in_order(self, requests):
        """Write request files one after the other, each stamped a second after the one before it, whatever the ids are:
        the way two commands run at once would leave them, with the order they were written in the files' times."""
        stamp = time.time_ns() - 60 * 10 ** 9
        paths = []
        for request_id, request in requests:
            path = store.write_request(self.out, {"pid": os.getpid()}, request, request_id)
            os.utime(path, ns=(stamp, stamp))
            stamp += 10 ** 9
            paths.append(path)
        return paths

    def answers(self, ids, *, seconds=15.0):
        """The answers to ``ids``, once every one is there (the event loop and the probe run meanwhile)."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            found = [store.read_json(store.response_path(self.out, request_id)) for request_id in ids]
            if all(found):
                return found
            self.app.processEvents()
            self.probe._tick()
            time.sleep(0.004)
        raise AssertionError(f"no answers to {ids}")

    def test_requests_are_handled_in_the_order_they_were_written(self):
        # A click and the eval written after it, both there when the probe first looks. The click runs from a timer, later
        # than the look, so an eval handled in the same look saw the button as it was; and the ids sort the eval first.
        for click_id, eval_id in (("a-click", "b-eval"), ("z-click", "y-eval")):
            with self.subTest(ids=(click_id, eval_id)):
                def build(window, layout):
                    button = QtWidgets.QPushButton("Toggle")
                    button.setCheckable(True)
                    layout.addWidget(button)
                window = self.make_window(f"Order window {click_id}", build)
                self.pump()
                select = {"window": f"Order window {click_id}", "widget": "Toggle"}
                self.write_in_order([
                    (eval_id + "-before", {"action": "eval", "expr": "widget.isChecked()", **select}),
                    (click_id, {"action": "click", "settle_ms": 0, **select}),
                    (eval_id, {"action": "eval", "expr": "widget.isChecked()", **select}),
                ])
                before, click, after = self.answers([eval_id + "-before", click_id, eval_id])
                self.assertEqual((before["result"]["value"], click["ok"], after["result"]["value"]), (False, True, True))
                self.assertLessEqual(before["wall_ms"], click["wall_ms"])
                self.assertLessEqual(click["wall_ms"], after["wall_ms"])
                window.close()

    def test_a_click_finds_the_window_the_click_before_it_opens(self):
        opened = []

        def build(window, layout):
            button = QtWidgets.QPushButton("Open")

            def open_second():
                second = self.make_window("Opened by the click", lambda inner, inner_layout: inner_layout.addWidget(
                    QtWidgets.QLabel("hello from the second window")))
                opened.append(second)
            button.clicked.connect(open_second)
            layout.addWidget(button)
        self.make_window("Opener", build)
        self.pump()
        self.write_in_order([
            ("first", {"action": "click", "window": "Opener", "widget": "Open", "settle_ms": 0}),
            ("second", {"action": "capture", "window": "Opened by the click"}),
        ])
        first, second = self.answers(["first", "second"])
        self.assertTrue(first["ok"], first)
        self.assertTrue(second["ok"], second)  # the window was there when the request was looked at

    def test_a_request_written_behind_one_that_is_still_being_written_waits_for_it(self):
        now = time.time_ns()
        slow = self.out / "requests" / store.request_name({"pid": os.getpid()}, "slow")
        slow.write_text('{"action": "pi', encoding="utf-8")
        os.utime(slow, ns=(now - 10 ** 8,) * 2)  # written first, and young enough that it may still be being written
        quick = store.write_request(self.out, {"pid": os.getpid()}, {"action": "ping"}, "quick")
        os.utime(quick, ns=(now - 5 * 10 ** 7,) * 2)
        self.pump(200)
        self.assertTrue(slow.exists() and quick.exists())  # neither is taken: the one written later does not go first
        slow.write_text('{"action": "ping"}', encoding="utf-8")
        os.utime(slow, ns=(now - 10 ** 8,) * 2)
        slow_answer, quick_answer = self.answers(["slow", "quick"])
        self.assertTrue(slow_answer["ok"] and quick_answer["ok"])
        self.assertLessEqual(slow_answer["wall_ms"], quick_answer["wall_ms"])

    def test_an_unknown_action_and_a_broken_request_get_answers_that_say_what_is_known(self):
        unknown = self.ask({"action": "levitate"})
        self.assertFalse(unknown["ok"])
        self.assertIn("click", unknown["detail"]["actions"])
        path = self.out / "requests" / store.request_name({"pid": os.getpid()}, "broken")
        path.write_text("{not json", encoding="utf-8")
        deadline = time.monotonic() + 5
        while not store.response_path(self.out, "broken").exists() and time.monotonic() < deadline:
            self.pump(30)
        self.assertFalse(store.read_json(store.response_path(self.out, "broken"))["ok"])


class DialogTests(ProbeCase):
    def test_a_modal_dialog_does_not_stop_the_probe_so_it_can_be_captured_and_answered(self):
        outcome = {}

        def build(window, layout):
            button = QtWidgets.QPushButton("Ask me")
            button.setObjectName("ask")

            def open_dialog():
                box = QtWidgets.QMessageBox(window)
                box.setWindowTitle("Modal question")
                box.setText("Proceed?")
                box.addButton("Yes please", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                box.addButton("No thanks", QtWidgets.QMessageBox.ButtonRole.RejectRole)
                box.exec()
                outcome["clicked"] = box.clickedButton().text()
            button.clicked.connect(open_dialog)
            layout.addWidget(button)
        self.make_window("Parent window", build)
        self.pump()
        agent = FileAgent(self.out, os.getpid())
        answers = {}

        def act():
            answers["click"] = agent.ask({"action": "click", "window": "Parent window", "widget": "ask", "block_ms": 300, "settle_ms": 0})
            answers["windows"] = agent.ask({"action": "windows"})
            answers["capture"] = agent.ask({"action": "capture", "window": "Modal question"})
            answers["tree"] = agent.ask({"action": "tree", "window": "Modal question"})
            answers["answer"] = agent.ask({"action": "click", "window": "Modal question", "widget": "Yes please", "settle_ms": 0})
        thread = threading.Thread(target=act)
        thread.start()
        deadline = time.monotonic() + 30
        while thread.is_alive() and time.monotonic() < deadline:
            self.app.processEvents()
            self.probe._tick()
            time.sleep(0.004)
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(answers["click"]["result"]["completed"])  # the click is still inside exec()
        modal = [entry for entry in answers["windows"]["result"]["windows"] if entry["title"] == "Modal question"]
        self.assertTrue(modal and modal[0]["modal"], answers["windows"])
        self.assertTrue(Path(answers["capture"]["result"]["path"]).is_file())
        labels = json.dumps(answers["tree"]["result"]["tree"])
        self.assertIn("Proceed?", labels)
        self.assertTrue(answers["answer"]["ok"], answers["answer"])
        self.assertEqual(outcome["clicked"], "Yes please")
        self.pump(200)
        self.assertTrue(any(event["ev"] == "action_completed" for event in self.events()))

    def test_a_request_written_behind_one_that_opens_a_modal_dialog_goes_ahead_once_that_one_is_answered(self):
        outcome = {}

        def build(window, layout):
            button = QtWidgets.QPushButton("Ask me")

            def open_dialog():
                box = QtWidgets.QMessageBox(window)
                box.setWindowTitle("Queued question")
                box.setText("Proceed?")
                box.addButton("Yes please", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                box.exec()
                clicked = box.clickedButton()
                outcome["clicked"] = clicked.text() if clicked is not None else None
            button.clicked.connect(open_dialog)
            layout.addWidget(button)
        self.make_window("Queued parent", build)
        self.pump()

        def close_if_stuck():  # a request that is held up for good must fail this test, not leave its dialog open for ever
            modal = QtWidgets.QApplication.activeModalWidget()
            if modal is not None:
                modal.reject()
        QtCore.QTimer.singleShot(10_000, close_if_stuck)
        stamp = time.time_ns() - 10 ** 9
        for index, (request_id, request) in enumerate((
                ("1-open", {"action": "click", "window": "Queued parent", "widget": "Ask me", "block_ms": 200, "settle_ms": 0}),
                ("2-answer", {"action": "click", "window": "Queued question", "widget": "Yes please", "settle_ms": 0}))):
            path = store.write_request(self.out, {"pid": os.getpid()}, request, request_id)
            os.utime(path, ns=(stamp + index * 10 ** 6,) * 2)
        deadline = time.monotonic() + 30
        paths = [store.response_path(self.out, name) for name in ("1-open", "2-answer")]
        while not all(path.exists() for path in paths) and time.monotonic() < deadline:
            self.app.processEvents()
            self.probe._tick()
            time.sleep(0.004)
        opened, answered = (store.read_json(path) for path in paths)
        self.assertFalse(opened["result"]["completed"])  # still inside exec() when its answer fell due
        self.assertTrue(answered["ok"], answered)       # and the request behind it was not held until the dialog was gone
        self.assertEqual(outcome["clicked"], "Yes please")

    def test_file_dialogs_take_their_answers_from_the_queue_and_cancel_when_it_is_empty(self):
        reply = self.ok({"action": "dialog_answers", "answers": [
            {"method": "getSaveFileName", "path": "{files}/saved.txt", "filter": "Text files (*.txt)"},
            {"path": "/some/folder"},
            {"paths": ["/a.fasta", "/b.fasta"], "method": "getOpenFileNames"},
            {"cancel": True},
        ]})
        self.assertEqual(reply["queued"], 4)
        files = str(self.out / "files")
        path, chosen_filter = QtWidgets.QFileDialog.getSaveFileName(None, "Save the output", "start", "Text files (*.txt);;All files (*)")
        self.assertEqual((path, chosen_filter), (f"{files}/saved.txt", "Text files (*.txt)"))
        self.assertEqual(QtWidgets.QFileDialog.getExistingDirectory(None, "Pick a folder"), "/some/folder")
        names, _ = QtWidgets.QFileDialog.getOpenFileNames(None, "Open")
        self.assertEqual(names, ["/a.fasta", "/b.fasta"])
        self.assertEqual(QtWidgets.QFileDialog.getOpenFileName(None, "Open one"), ("", ""))  # the queued cancel
        self.assertEqual(QtWidgets.QFileDialog.getOpenFileName(None, "Open another"), ("", ""))  # nothing queued
        calls = self.events("file_dialog")
        self.assertEqual([call["function"] for call in calls],
                         ["getSaveFileName", "getExistingDirectory", "getOpenFileNames", "getOpenFileName", "getOpenFileName"])
        self.assertEqual(calls[0]["caption"], "Save the output")
        self.assertEqual(calls[0]["directory"], "start")
        self.assertEqual([call["queued"] for call in calls], [True, True, True, True, False])
        self.assertEqual(self.ok({"action": "dialog_answers"})["queued"], 0)
        self.probe.shutdown()
        self.assertNotIn("FileDialogAnswers", getattr(QtWidgets.QFileDialog.getSaveFileName, "__qualname__", ""))

    def test_desktop_open_calls_are_recorded_and_still_go_through(self):
        from utilities import Desktop_Open
        self.probe.desktop_open.restore()
        calls = []
        with mock.patch.object(Desktop_Open, "open_path", lambda path: calls.append(("path", path)) or True), \
                mock.patch.object(Desktop_Open, "open_url", lambda url, new_tab=False: calls.append(("url", url, new_tab)) or True):
            fake_path, fake_url = Desktop_Open.open_path, Desktop_Open.open_url
            self.probe.desktop_open.refresh()
            self.assertIsNot(Desktop_Open.open_path, fake_path)
            self.assertTrue(Desktop_Open.open_path(os.path.join("some", "folder")))
            self.assertTrue(Desktop_Open.open_url("https://example.org/page", new_tab=True))
            self.probe.desktop_open.restore()
            self.assertIs(Desktop_Open.open_path, fake_path)
            self.assertIs(Desktop_Open.open_url, fake_url)
        self.assertEqual(calls, [("path", os.path.join("some", "folder")), ("url", "https://example.org/page", True)])
        recorded = self.events("desktop_open")
        self.assertEqual([(event["function"], event["target"]) for event in recorded],
                         [("open_path", os.path.join("some", "folder")), ("open_url", "https://example.org/page")])
        self.assertEqual(recorded[1]["options"], {"new_tab": True})
        self.assertTrue(recorded[0]["result"])


class SafetyTests(ProbeCase):
    def test_a_failure_inside_the_probe_is_logged_and_turns_it_off_without_reaching_the_program(self):
        with mock.patch.object(self.probe.windows, "scan", side_effect=RuntimeError("scan blew up")):
            self.probe._tick()  # must not raise
        self.assertTrue(self.recorder.disabled)
        self.assertIn("scan blew up", (self.recorder.folder / "errors.log").read_text(encoding="utf-8"))
        self.assertTrue(self.events("probe_error"))
        self.assertTrue(self.events("probe_disabled"))
        count = len(self.events())
        self.make_window("After failure")
        self.pump(100)
        self.assertEqual(len(self.events()), count)  # nothing more is recorded
        request_id = "after"
        store.write_request(self.out, {"pid": os.getpid()}, {"action": "windows"}, request_id)
        self.pump(150)
        self.assertIsNone(store.read_json(store.response_path(self.out, request_id)))  # and nothing is answered

    def test_turning_the_probe_off_puts_back_what_it_replaced(self):
        names = ("getSaveFileName", "getOpenFileName", "getExistingDirectory")
        replaced = {name: getattr(QtWidgets.QFileDialog, name) for name in names}
        self.probe.shutdown()
        for name in names:
            self.assertIsNot(getattr(QtWidgets.QFileDialog, name), replaced[name])  # the originals are back
            self.assertTrue(callable(getattr(QtWidgets.QFileDialog, name)))

    def test_removing_the_gate_file_turns_the_probe_off_within_a_second_or_so(self):
        store.remove_gate(self.root)
        for _ in range(self.runtime.GATE_EVERY * (self.runtime.GATE_MISSES - 1) + 1):
            self.probe._tick()
        self.assertFalse(self.recorder.disabled)  # one look that finds it gone, or two, is not enough
        for _ in range(self.runtime.GATE_EVERY):
            self.probe._tick()
        self.assertTrue(self.recorder.disabled)
        self.assertEqual(self.events("probe_disabled")[-1]["reason"], "the gate file was removed")
        self.assertTrue(self.probe._stopped)

    def test_the_probe_opens_no_window_of_its_own_and_adds_no_widget_to_the_program(self):
        self.make_window("Only window")
        before = {id(widget) for widget in self.app.allWidgets()}
        self.pump(200)
        self.ok({"action": "windows"})
        self.ok({"action": "tree", "window": "Only window"})
        self.ok({"action": "capture", "window": "Only window"})
        after = [widget for widget in self.app.allWidgets() if id(widget) not in before]
        self.assertEqual([widget for widget in after if widget.isWindow()], [])

    def test_the_probe_does_not_keep_a_window_alive(self):
        import gc
        import weakref
        window = QtWidgets.QWidget()
        window.setWindowTitle("Weak window")
        window.show()
        self.pump(100)
        reference = weakref.ref(window)
        window.hide()
        del window
        self.pump(100)
        gc.collect()
        self.assertIsNone(reference())


# ---------------------------------------------------------------------------------------------
# The command line, against a probe running here
# ---------------------------------------------------------------------------------------------

class CommandLineTests(ProbeCase):
    def run_cli(self, arguments):
        """Run the command line in a thread while this thread runs the event loop, as the agent and the program do."""
        result = {}

        def work():
            result["code"], result["text"] = run_json(["--checkout", str(self.root), *arguments])
        thread = threading.Thread(target=work)
        thread.start()
        deadline = time.monotonic() + 30
        while thread.is_alive() and time.monotonic() < deadline:
            self.app.processEvents()
            self.probe._tick()
            time.sleep(0.004)
        thread.join(5)
        self.assertFalse(thread.is_alive(), arguments)
        return result["code"], result["text"]

    def test_do_sends_a_request_by_kind_and_prints_the_answer(self):
        self.make_window("Do window")
        self.pump()
        code, text = self.run_cli(["do", "--kind", "tools", '{"action": "windows"}'])
        self.assertEqual(code, 0, text)
        answer = json.loads(text)
        self.assertTrue(answer["ok"])
        self.assertIn("Do window", [entry["title"] for entry in answer["result"]["windows"]])
        self.assertEqual((answer["pid"], answer["kind"]), (os.getpid(), "tools"))

    def test_do_takes_an_action_with_key_value_pairs_and_a_pid(self):
        clicks = []

        def build(window, layout):
            button = QtWidgets.QPushButton("Press")
            button.clicked.connect(lambda: clicks.append(1))
            layout.addWidget(button)
        self.make_window("Pairs window", build)
        self.pump()
        code, text = self.run_cli(["do", "--pid", str(os.getpid()), "click", "window=Pairs window", "widget=Press", "settle_ms=0"])
        self.assertEqual(code, 0, text)
        self.assertEqual(len(clicks), 1)
        code, text = self.run_cli(["do", "--kind", "tools", "click", "window=Pairs window", "widget=Missing", "settle_ms=0"])
        self.assertEqual(code, 1)  # an error answer is a failing exit code
        self.assertFalse(json.loads(text)["ok"])

    def test_wait_finds_a_window_by_title_and_waits_for_it_to_go(self):
        self.make_window("Waited window")
        self.pump(120)
        code, text = self.run_cli(["wait", "--kind", "tools", "--title", "^waited", "--timeout", "10"])
        self.assertEqual(code, 0, text)
        self.assertEqual(json.loads(text)["window"]["title"], "Waited window")
        code, _ = self.run_cli(["wait", "--kind", "tools", "--title", "No such window", "--timeout", "0.3"])
        self.assertEqual(code, 1)
        self.made[-1].hide()
        self.pump(120)
        code, text = self.run_cli(["wait", "--kind", "tools", "--title", "^waited", "--gone", "--timeout", "10"])
        self.assertEqual((code, text.strip()), (0, "gone"))

    def test_step_stamps_the_events_and_captures_that_follow(self):
        code, text = self.run_cli(["step", "open the console"])
        self.assertEqual(code, 0, text)
        self.assertIn("step 1: open the console", text)
        self.make_window("Stamped window")
        self.pump(150)
        shown = next(event for event in self.events("window_shown") if event["window"]["title"] == "Stamped window")
        self.assertEqual(shown["step"], "open the console")
        self.assertTrue(list((self.out / "captures").glob("*open-the-console-Stamped-window.png")))

    def test_status_lists_what_is_on_and_running(self):
        code, text = self.run_cli(["status", "--json"])
        status = json.loads(text)
        self.assertTrue(status["on"])
        self.assertEqual(Path(status["out"]), self.out)
        self.assertTrue(any(process["pid"] == os.getpid() and process["alive"] for process in status["processes"]))

    def test_do_without_a_running_process_says_which_are_known(self):
        with self.assertRaises(SystemExit) as caught:
            probe_cli.main(["--checkout", str(self.root), "do", "--kind", "viewer", "--timeout", "0.2", "windows"])
        self.assertIn("No running process matches", str(caught.exception))
        self.assertIn("tools", str(caught.exception))

    def test_a_request_may_be_json_a_file_or_stdin(self):
        self.assertEqual(probe_cli.parse_request(['{"action": "tree", "window": "x"}']), {"action": "tree", "window": "x"})
        path = self.root / "request.json"
        path.write_text('{"action": "ping"}', encoding="utf-8")
        self.assertEqual(probe_cli.parse_request([f"@{path}"]), {"action": "ping"})
        with mock.patch.object(sys, "stdin", io.StringIO('{"action": "windows"}')):
            self.assertEqual(probe_cli.parse_request(["-"]), {"action": "windows"})
        self.assertEqual(probe_cli.parse_request(["click", "widget=Save & Run", "count=2", 'modifiers=["ctrl"]']),
                         {"action": "click", "widget": "Save & Run", "count": 2, "modifiers": ["ctrl"]})
        for bad in (["{}"], ["Click"], ["click", "novalue"]):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                probe_cli.parse_request(bad)


# ---------------------------------------------------------------------------------------------
# The timeline and the contact sheet
# ---------------------------------------------------------------------------------------------

def fake_process(out, pid, kind, boot_ms, events, *, ancestors=(), detail=None, argv=("x",), created=None):
    folder = Path(out) / "procs" / f"{pid}-{kind}"
    folder.mkdir(parents=True)
    record = {"pid": pid, "ppid": ancestors[0] if ancestors else 1, "kind": kind, "detail": detail, "argv": list(argv),
              "boot_wall_ms": boot_ms, "ancestors": list(ancestors)}
    if created is not None:
        record["created"] = created
    store.atomic_write_json(folder / "process.json", record)
    lines = [json.dumps({"ev": name, "wall_ms": when, "mono_ms": when, "pid": pid, "kind": kind, "step": "", **fields})
             for name, when, fields in events]
    (folder / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


class SheetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = store.ensure_out(Path(self.temp.name) / "out")

    @staticmethod
    def window(title, cls="QWidget"):
        return {"class": cls, "title": title, "geometry": [0, 0, 100, 100]}

    def test_the_timeline_joins_the_launch_chain_and_gives_the_times_a_round_is_read_for(self):
        base = 1_700_000_000_000
        splash = self.window("", "SplashWindow")
        config = self.window("EMAP-SSN Configuration", "ConfigGUI")
        fake_process(self.out, 100, "splash", base + 20, [
            ("process_start", base + 20, {}), ("app_ready", base + 320, {}),
            ("window_shown", base + 330, {"serial": 1, "window": splash, "late": True}),
            ("window_first_painted", base + 350, {"serial": 1, "window": splash}),
            ("window_hidden", base + 3400, {"serial": 1, "window": splash}),
            ("exit", base + 3450, {})], detail="viewer", created=base / 1000)
        fake_process(self.out, 200, "monitor", base + 500, [("process_start", base + 500, {}), ("exit", base + 3600, {})],
                     ancestors=[99, 100], created=(base + 480) / 1000)  # 99: the venv's launcher stub, which has no record
        fake_process(self.out, 300, "config", base + 700, [
            ("process_start", base + 700, {}), ("app_ready", base + 3200, {}),
            ("window_shown", base + 3210, {"serial": 1, "window": config}),
            ("window_first_painted", base + 3260, {"serial": 1, "window": config}),
            ("exit", base + 3500, {})], ancestors=[250, 200], created=(base + 650) / 1000)
        (launch,) = sheet.launches(self.out)
        self.assertEqual((launch["root"], launch["pid"]), ("splash viewer", 100))
        self.assertEqual([entry["kind"] for entry in launch["processes"]], ["splash", "monitor", "config"])
        self.assertEqual(sheet.summary_line(launch), "splash shown +330 ms (seen by a scan), first painted +350 ms, "
                                                     "config window shown +3210 ms, first painted +3260 ms, splash closed +3400 ms")
        text = sheet.render_timeline(self.out)
        self.assertIn("Launch 1: splash viewer (pid 100)", text)
        self.assertIn("monitor process started", text)
        self.assertRegex(text, r"3210 ms\s+config: shown ConfigGUI 'EMAP-SSN Configuration'")

    def test_a_picture_made_another_way_is_attached_under_the_current_step_and_shown_on_the_sheet(self):
        import struct
        import zlib

        def tiny_png(width, height):
            def chunk(kind, data):
                return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
            rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
            return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                    + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))

        image = Path(self.temp.name) / "canvas.png"
        image.write_bytes(tiny_png(30, 20))
        store.write_step(self.out, "the canvas by print")
        code, text = run_json(["--out", str(self.out), "attach", str(image), "--caption", "Viewer canvas <print>"])
        self.assertEqual(code, 0, text)
        (stored,) = list((self.out / "captures").glob("attached-0001-the-canvas-by-print-*.png"))
        self.assertEqual(stored.read_bytes(), image.read_bytes())
        page = sheet.contact_sheet(self.out)
        self.assertIn(f"src='captures/{stored.name}'", page)
        self.assertIn("30&times;20 px", page)
        self.assertIn("Viewer canvas &lt;print&gt;", page)
        self.assertIn("the canvas by print", page)
        broken = Path(self.temp.name) / "not.png"
        broken.write_bytes(b"not a picture at all, really")
        with self.assertRaises(SystemExit):
            probe_cli.main(["--out", str(self.out), "attach", str(broken)])

    def test_two_launches_are_two_entries_and_an_empty_run_says_so(self):
        self.assertIn("No process has recorded anything", sheet.render_timeline(self.out))
        fake_process(self.out, 11, "splash", 1000, [("process_start", 1000, {})])
        fake_process(self.out, 12, "splash", 9000, [("process_start", 9000, {})])
        self.assertEqual(len(sheet.launches(self.out)), 2)

    def test_the_contact_sheet_lists_every_capture_in_order_grouped_by_step_with_relative_links(self):
        base = 1_700_000_000_000
        steps = [{"seq": 1, "label": "first step", "wall_ms": base}, {"seq": 2, "label": "second <step>", "wall_ms": base + 1000},
                 {"seq": 3, "label": "empty step", "wall_ms": base + 5000}]
        (self.out / "steps.jsonl").write_text("".join(json.dumps(step) + "\n" for step in steps), encoding="utf-8")

        def capture(when, title, number, **extra):
            return ("capture", when, {"file": f"captures/shot-{number}.png", "path": "x", "width": 800, "height": 600, "dpr": 2.0,
                                      "logical": [400, 300], "title": title, "window": title, "reason": "auto", "number": number,
                                      "screen": {"name": "DP-1", "scale": 2.0}, "flat_colour": None, "drawn_elsewhere": [], **extra})
        fake_process(self.out, 10, "config", base, [
            ("process_start", base, {}),
            capture(base + 100, "Config <b>window</b>", 1),
            ("dialog_shown", base + 1200, {"serial": 1, "dialog": {"class": "QMessageBox", "title": "Careful", "text": "Sure?", "buttons": ["Yes", "No"]}}),
            capture(base + 1300, "Console", 2, flat_colour="#000000", drawn_elsewhere=[{"class": "QOpenGLWidget", "flat_colour": "#000000"}]),
            ("desktop_open", base + 1400, {"function": "open_path", "target": "/logs", "result": True}),
        ])
        page = sheet.contact_sheet(self.out)
        for part in ("first step", "second &lt;step&gt;", "empty step", "No capture in this step."):
            self.assertIn(part, page)
        self.assertLess(page.index("first step"), page.index("second &lt;step&gt;"))
        self.assertLess(page.index("second &lt;step&gt;"), page.index("empty step"))
        self.assertIn("src='captures/shot-1.png'", page)
        self.assertIn("src='captures/shot-2.png'", page)
        self.assertLess(page.index("shot-1.png"), page.index("second &lt;step&gt;"))  # step 1's capture is in step 1
        self.assertGreater(page.index("shot-2.png"), page.index("second &lt;step&gt;"))
        self.assertNotIn("<b>window</b>", page)  # titles are escaped
        self.assertIn("DPR 2.0", page)
        self.assertIn("800&times;600 px", page)
        self.assertIn("DP-1", page)
        self.assertIn("one colour (#000000)", page)
        self.assertIn("QOpenGLWidget came out one colour", page)
        self.assertIn("dialog QMessageBox", page)
        self.assertIn("open_path(", page)
        self.assertNotIn("http://", page)  # self-contained: nothing is loaded from outside


if __name__ == "__main__":
    unittest.main()
