# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Install_Dependencies: ESM/Transformers stack fingerprinting and recorded
import-validation passes.

Nothing here installs anything: fake venv folders are created in temp dirs and
uv, pip and the validators are mocked.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import Detect_GPU  # noqa: E402
import Install_Dependencies  # noqa: E402
from tests.dependency_install_fixtures import saved_state  # noqa: E402
from tests.gpu_fixtures import detected_report  # noqa: E402


class _AuditedOpens:
    """Collect the files opened for reading while `active` is true.

    sys.addaudithook cannot be undone, so one hook stays for the life of the
    test process and does nothing unless a test switches it on.
    """

    active = False
    paths: list[str] = []
    installed = False

    @classmethod
    def install(cls):
        if cls.installed:
            return
        cls.installed = True

        def hook(event, args):
            if cls.active and event == "open" and isinstance(args[0], (str, os.PathLike)):
                cls.paths.append(os.fspath(args[0]))

        sys.addaudithook(hook)


def _fake_venv(root, *, layout="windows", packages=None):
    """A venv-shaped folder: site-packages with dist-info folders, an interpreter, a pyvenv.cfg.

    Returns (venv, python). Nothing in it is importable; the fingerprint only stats it.
    """
    venv = Path(root) / "venv"
    base = Path(root) / "base_python"
    base.mkdir()
    (base / ("python.exe" if layout == "windows" else "python3")).write_bytes(b"base interpreter")
    if layout == "windows":
        site = venv / "Lib" / "site-packages"
        python = venv / "Scripts" / "python.exe"
    else:
        site = venv / "lib" / "python3.13" / "site-packages"
        python = venv / "bin" / "python"
    site.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"venv interpreter")
    (venv / "pyvenv.cfg").write_text(
        f"home = {base}\nimplementation = CPython\nversion_info = 3.13.14\n", encoding="utf-8"
    )
    for name, version in (packages or {
        "esm": "3.4.1.post1", "transformers": "5.17.0", "torch": "2.14.0+cu132", "scipy": "1.17.0",
    }).items():
        dist = site / f"{name}-{version}.dist-info"
        dist.mkdir()
        (dist / "RECORD").write_text(f"{name}/__init__.py,sha256=x,10\n", encoding="utf-8")
        package = site / name
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
    (site / "typing_extensions.py").write_text("# single-file module\n", encoding="utf-8")
    return venv, python


def _site_packages(venv):
    for candidate in (venv / "Lib" / "site-packages", *sorted((venv / "lib").glob("python*/site-packages"))):
        if candidate.is_dir():
            return candidate
    raise AssertionError("fake venv has no site-packages")


class EsmStackFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def _fingerprint(self, venv, python):
        return Install_Dependencies.esm_stack_fingerprint(venv, python)

    def test_the_fingerprint_is_stable_and_works_for_both_venv_layouts(self):
        for layout in ("windows", "posix"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as root:
                venv, python = _fake_venv(root, layout=layout)
                first = self._fingerprint(venv, python)
                self.assertRegex(first, r"^[0-9a-f]{64}$")
                self.assertEqual(first, self._fingerprint(venv, python))

    def test_no_site_packages_means_unknown_not_a_fingerprint(self):
        self.assertIsNone(self._fingerprint(Path(self.temp.name) / "missing", Path("python")))
        empty = Path(self.temp.name) / "empty"
        empty.mkdir()
        self.assertIsNone(self._fingerprint(empty, empty / "python"))

    def test_bytecode_folders_appearing_are_not_a_change(self):
        # A pass writes .pyc files; running any program later writes more. Neither
        # may make the environment look changed.
        venv, python = _fake_venv(self.temp.name)
        before = self._fingerprint(venv, python)
        site = _site_packages(venv)
        (site / "__pycache__").mkdir()
        (site / "__pycache__" / "typing_extensions.cpython-313.pyc").write_bytes(b"pyc")
        (site / "torch" / "__pycache__").mkdir()
        (site / "torch" / "__pycache__" / "x.pyc").write_bytes(b"pyc")
        (site / "torch" / "new_submodule.py").write_text("", encoding="utf-8")
        self.assertEqual(before, self._fingerprint(venv, python))

    def test_it_stats_files_and_opens_only_pyvenv_cfg(self):
        # Antivirus scans every file a program opens: the fingerprint must open
        # none of the packages' files, nor the RECORD files.
        _AuditedOpens.install()
        venv, python = _fake_venv(self.temp.name)
        _AuditedOpens.paths = []
        _AuditedOpens.active = True
        try:
            self._fingerprint(venv, python)
        finally:
            _AuditedOpens.active = False
        ours = [Path(path) for path in _AuditedOpens.paths if Path(path).is_relative_to(Path(self.temp.name))]
        self.assertEqual(ours, [venv / "pyvenv.cfg"])

    def test_the_interpreter_path_counts_but_its_letter_case_does_not_on_windows(self):
        venv, python = _fake_venv(self.temp.name)
        before = self._fingerprint(venv, python)
        # The same venv reached by another interpreter path is another environment...
        other = python.with_name("python3.exe")
        shutil.copy2(python, other)  # same size and modification time: only the path differs
        self.assertNotEqual(before, self._fingerprint(venv, other))
        # ...but where paths ignore case, launchers that spell the folder differently agree.
        if os.path.normcase("A") == "a":
            self.assertEqual(before, self._fingerprint(venv, Path(str(python).swapcase())))

    def test_the_fingerprint_of_300_packages_takes_under_250_ms_on_average(self):
        # Milliseconds: 300 packages, as in the real environment.
        packages = {f"package{index}": "1.0" for index in range(300)}
        venv, python = _fake_venv(self.temp.name, packages=packages)
        start = time.perf_counter()
        for _ in range(10):
            self._fingerprint(venv, python)
        self.assertLess((time.perf_counter() - start) / 10, 0.25)


class EsmStackRecordTests(unittest.TestCase):
    """The recorded pass of the ESM/Transformers import test, and when it is skipped."""

    PASSED = "ESM/Transformers import validation: passed"
    SKIPPED = "ESM/Transformers import validation: unchanged since the last check (skipped)"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.report = detected_report(
            backend_candidates=[{"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]}],
            compatibility_revision=Detect_GPU.COMPATIBILITY_REVISION,
            platform="windows",
            os="Windows",
            devices=[],
        )
        self.spec = Install_Dependencies.backend_specs(self.report)[0]
        self._fresh(self.temp.name)

    def _fresh(self, root):
        """A new fake venv under `root` with the backend state an earlier install saved."""
        self.base = Path(root) / "base_python"
        self.venv, self.python = _fake_venv(root)
        self.state_path = self.venv / Install_Dependencies.STATE_FILENAME
        Install_Dependencies.write_state(
            self.state_path,
            saved_state(self.spec, ROOT / "src" / "requirements.txt"),
            self.report,
        )
        self.record_path = self.venv / Install_Dependencies.ESM_STACK_RECORD_FILENAME

    def _check(self, esm_ok=True):
        """One launcher check against the fake venv; returns (ready, stdout lines, smoke-test mock)."""
        stdout = io.StringIO()
        pinned = lambda _python, package: (  # noqa: E731
            Install_Dependencies.TRANSFORMERS_VERSION
            if package == "transformers" else Install_Dependencies.ESM_VERSION
        )
        with mock.patch.object(Install_Dependencies, "venv_python", return_value=self.python), \
                mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value=self.report), \
                mock.patch.object(
                    Install_Dependencies, "validate_backend",
                    return_value={"validated_devices": [{"spec": "cpu", "success": True}]},
                ), \
                mock.patch.object(Install_Dependencies, "_installed_version", side_effect=pinned), \
                mock.patch.object(Install_Dependencies, "validate_package_consistency", return_value=True), \
                mock.patch.object(Install_Dependencies, "validate_esm_stack", return_value=esm_ok) as smoke, \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            ready = Install_Dependencies.environment_is_ready(
                project_root=ROOT, venv=self.venv, uv_executable="uv"
            )
        return ready, stdout.getvalue().splitlines(), smoke

    def test_a_pass_is_recorded_and_the_next_check_skips_the_test(self):
        ready, lines, smoke = self._check()
        self.assertTrue(ready)
        smoke.assert_called_once_with(self.python)
        self.assertIn(self.PASSED, lines)
        record = json.loads(self.record_path.read_text(encoding="utf-8"))
        self.assertEqual(
            record["fingerprint"], Install_Dependencies.esm_stack_fingerprint(self.venv, self.python)
        )

        ready, lines, smoke = self._check()
        self.assertTrue(ready)
        smoke.assert_not_called()
        self.assertIn(self.SKIPPED, lines)
        self.assertNotIn(self.PASSED, lines)
        self.assertIn("Dependency environment is ready (CPU).", "\n".join(lines))

    def test_any_change_the_test_depends_on_reruns_it(self):
        site = lambda venv: _site_packages(venv)  # noqa: E731

        def upgrade_package(venv, python):
            old = site(venv) / "scipy-1.17.0.dist-info"
            old.rename(old.with_name("scipy-1.17.1.dist-info"))

        def add_package(venv, python):
            (site(venv) / "pandas-3.0.0.dist-info").mkdir()

        def remove_package(venv, python):
            old = site(venv) / "scipy-1.17.0.dist-info"
            (old / "RECORD").unlink()
            old.rmdir()

        def rewrite_record(venv, python):
            (site(venv) / "torch-2.14.0+cu132.dist-info" / "RECORD").write_text(
                "a much longer RECORD than before\n" * 5, encoding="utf-8"
            )

        def touch_record(venv, python):
            record = site(venv) / "torch-2.14.0+cu132.dist-info" / "RECORD"
            status = record.stat()
            os.utime(record, ns=(status.st_atime_ns, status.st_mtime_ns + 5_000_000_000))

        def add_top_level_package(venv, python):
            (site(venv) / "stray_package").mkdir()

        def edit_single_file_module(venv, python):
            (site(venv) / "typing_extensions.py").write_text("# edited, and longer\n" * 3, encoding="utf-8")

        def replace_interpreter(venv, python):
            python.write_bytes(b"a different venv interpreter")

        def upgrade_base_python(venv, python):
            base = next(self.base.iterdir())
            status = base.stat()
            os.utime(base, ns=(status.st_atime_ns, status.st_mtime_ns + 5_000_000_000))

        def edit_pyvenv_cfg(venv, python):
            with (venv / "pyvenv.cfg").open("a", encoding="utf-8") as handle:
                handle.write("version_info = 3.13.15\n")

        mutations = {
            "a package version": upgrade_package,
            "a package added": add_package,
            "a package removed": remove_package,
            "a RECORD's size": rewrite_record,
            "a RECORD's modification time": touch_record,
            "a top-level package folder": add_top_level_package,
            "a single-file module": edit_single_file_module,
            "the venv interpreter": replace_interpreter,
            "the Python it runs on": upgrade_base_python,
            "pyvenv.cfg": edit_pyvenv_cfg,
        }
        for label, mutate in mutations.items():
            with self.subTest(changed=label), tempfile.TemporaryDirectory() as root:
                self._fresh(root)
                self.assertTrue(self._check()[0])  # records
                self.assertFalse(self._check()[2].called)  # unchanged: skipped
                mutate(self.venv, self.python)
                ready, lines, smoke = self._check()
                self.assertTrue(ready)
                smoke.assert_called_once()
                self.assertIn(self.PASSED, lines)

    def test_changing_the_test_program_reruns_it(self):
        self.assertTrue(self._check()[0])
        self.assertFalse(self._check()[2].called)
        with mock.patch.object(
            Install_Dependencies, "_esm_stack_program", return_value="import esm  # edited"
        ):
            self.assertFalse(Install_Dependencies.esm_stack_validation_is_current(self.venv, self.python))
            ready, lines, smoke = self._check()
        self.assertTrue(ready)
        smoke.assert_called_once()
        self.assertIn(self.PASSED, lines)

    def test_changing_the_esm_or_transformers_pin_reruns_it(self):
        # The program text embeds both pins, so hold it fixed: the pins count on their own.
        with mock.patch.object(Install_Dependencies, "_esm_stack_program", return_value="import esm"):
            recorded = Install_Dependencies.esm_stack_fingerprint(self.venv, self.python)
            for name in ("ESM_VERSION", "TRANSFORMERS_VERSION"):
                with self.subTest(pin=name), mock.patch.object(Install_Dependencies, name, "9.9.9"):
                    self.assertNotEqual(
                        Install_Dependencies.esm_stack_fingerprint(self.venv, self.python), recorded
                    )
        # And with the real program text, a pin changing is not current any more.
        self.assertTrue(self._check()[0])
        self.assertFalse(self._check()[2].called)
        for name in ("ESM_VERSION", "TRANSFORMERS_VERSION"):
            with self.subTest(pin=name, flow=True), mock.patch.object(Install_Dependencies, name, "9.9.9"):
                self.assertFalse(Install_Dependencies.esm_stack_validation_is_current(self.venv, self.python))

    def test_a_failed_test_keeps_no_record(self):
        ready, lines, smoke = self._check(esm_ok=False)
        self.assertFalse(ready)
        smoke.assert_called_once()
        self.assertFalse(self.record_path.exists())
        self.assertNotIn(self.PASSED, lines)

    def test_a_failure_after_a_change_removes_the_old_record(self):
        self.assertTrue(self._check()[0])
        self.assertTrue(self.record_path.exists())
        (_site_packages(self.venv) / "scipy-1.17.0.dist-info").rename(
            _site_packages(self.venv) / "scipy-1.17.1.dist-info"
        )
        ready, _, smoke = self._check(esm_ok=False)
        self.assertFalse(ready)
        smoke.assert_called_once()
        self.assertFalse(self.record_path.exists())
        # ...so once it works again, the test runs again rather than trusting an old pass.
        ready, lines, smoke = self._check()
        self.assertTrue(ready)
        smoke.assert_called_once()

    def test_a_corrupt_or_old_record_reruns_the_test_without_touching_the_saved_state(self):
        current = Install_Dependencies.esm_stack_fingerprint(self.venv, self.python)
        records = {
            "missing": None,
            "empty": "",
            "not JSON": "{not json",
            "a list": "[]",
            "no fingerprint": json.dumps({"schema": Install_Dependencies.ESM_STACK_RECORD_SCHEMA}),
            "a fingerprint that is not text": json.dumps(
                {"schema": Install_Dependencies.ESM_STACK_RECORD_SCHEMA, "fingerprint": 5}
            ),
            "a stale fingerprint": json.dumps(
                {"schema": Install_Dependencies.ESM_STACK_RECORD_SCHEMA, "fingerprint": "0" * 64}
            ),
            "another schema": json.dumps(
                {"schema": Install_Dependencies.ESM_STACK_RECORD_SCHEMA + 1, "fingerprint": current}
            ),
            "binary garbage": b"\xff\xfe\x00\x80",
        }
        state_before = self.state_path.read_bytes()
        for label, content in records.items():
            with self.subTest(record=label):
                if content is None:
                    if self.record_path.exists():
                        self.record_path.unlink()
                elif isinstance(content, bytes):
                    self.record_path.write_bytes(content)
                else:
                    self.record_path.write_text(content, encoding="utf-8")
                ready, lines, smoke = self._check()
                self.assertTrue(ready)
                smoke.assert_called_once()
                self.assertIn(self.PASSED, lines)
                # The check never re-syncs: the saved backend state is byte-for-byte as it was.
                self.assertEqual(self.state_path.read_bytes(), state_before)
                json.loads(self.record_path.read_text(encoding="utf-8"))  # and the record is good again

    def test_a_state_saved_before_the_record_existed_is_still_ready(self):
        # The record is a file of its own, not a field of ssn_backend.json, so an
        # install made before this change has nothing to mismatch and no re-sync.
        state = Install_Dependencies.read_state(self.state_path)
        self.assertFalse([key for key in state if "esm_stack" in key])
        specs = Install_Dependencies.backend_specs(self.report)
        fingerprint = Install_Dependencies.hardware_fingerprint(self.report)
        self.assertEqual(
            Install_Dependencies._state_mismatches(
                state, specs, fingerprint, ROOT / "src" / "requirements.txt"
            ),
            [],
        )
        self.assertFalse(self.record_path.exists())
        ready, lines, smoke = self._check()
        self.assertTrue(ready)
        smoke.assert_called_once()  # only the import test reruns
        self.assertNotIn("State mismatch", "\n".join(lines))

    def test_a_record_that_cannot_be_written_still_lets_the_launch_proceed(self):
        with mock.patch.object(Install_Dependencies.os, "replace", side_effect=PermissionError("read-only")):
            ready, lines, smoke = self._check()
        self.assertTrue(ready)
        smoke.assert_called_once()
        self.assertFalse(self.record_path.exists())
        self.assertEqual([path.name for path in self.venv.glob("*.tmp")], [])
        # With nothing recorded, the next launch simply runs the test again.
        self.assertFalse(Install_Dependencies.esm_stack_validation_is_current(self.venv, self.python))

    def test_a_venv_whose_fingerprint_is_unknown_never_skips(self):
        shutil.rmtree(_site_packages(self.venv))
        ready, _, smoke = self._check()
        self.assertTrue(ready)
        smoke.assert_called_once()
        self.assertFalse(self.record_path.exists())
        ready, _, smoke = self._check()
        smoke.assert_called_once()

    def _install(self, esm_ok=True, saved_state=True):
        if not saved_state:
            self.state_path.unlink()
        success = mock.Mock(returncode=0, stdout="", stderr="")
        validation = {"validated_devices": [{"spec": "cpu", "success": True}]}
        pinned = lambda _python, package: (  # noqa: E731
            Install_Dependencies.TRANSFORMERS_VERSION
            if package == "transformers" else Install_Dependencies.ESM_VERSION
        )
        with mock.patch.object(Install_Dependencies, "venv_python", return_value=self.python), \
                mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value=self.report), \
                mock.patch.object(Install_Dependencies, "_run", return_value=success), \
                mock.patch.object(Install_Dependencies, "validate_backend", return_value=validation), \
                mock.patch.object(Install_Dependencies, "install_backend", return_value=validation), \
                mock.patch.object(Install_Dependencies, "_installed_version", side_effect=pinned), \
                mock.patch.object(Install_Dependencies, "validate_package_consistency", return_value=True), \
                mock.patch.object(Install_Dependencies, "validate_esm_stack", return_value=esm_ok) as smoke, \
                mock.patch.object(Install_Dependencies, "write_state"), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = Install_Dependencies.install(project_root=ROOT, venv=self.venv, uv_executable="uv")
        return code, smoke

    def test_setup_records_a_passing_test_so_the_first_launch_after_it_skips(self):
        code, smoke = self._install()
        self.assertEqual(code, 0)
        self.assertTrue(smoke.called)
        self.assertTrue(Install_Dependencies.esm_stack_validation_is_current(self.venv, self.python))
        ready, lines, smoke = self._check()
        self.assertTrue(ready)
        smoke.assert_not_called()
        self.assertIn(self.SKIPPED, lines)

    def test_a_fresh_setup_records_the_pass_it_ends_with(self):
        # No saved state: nothing is reused, the ESM packages are installed, and the
        # test that ends the setup is the one that records.
        code, smoke = self._install(saved_state=False)
        self.assertEqual(code, 0)
        smoke.assert_called_once()
        self.assertTrue(Install_Dependencies.esm_stack_validation_is_current(self.venv, self.python))

    def test_setup_whose_test_fails_keeps_no_record(self):
        Install_Dependencies.record_esm_stack_validation(self.venv, self.python)
        self.assertTrue(self.record_path.exists())
        code, smoke = self._install(esm_ok=False)
        self.assertNotEqual(code, 0)
        self.assertFalse(self.record_path.exists())


if __name__ == "__main__":
    unittest.main()
