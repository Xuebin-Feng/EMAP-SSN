# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Install_Dependencies: backend profiles, validator programs and report
parsing, `uv pip check` consistency, requirement pins, and supported hosts.

Nothing here installs anything: uv, pip and the validators are mocked, except
that the validator programs run in a real interpreter against a fake torch.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import Detect_GPU  # noqa: E402
import Install_Dependencies  # noqa: E402
from tests.dependency_install_fixtures import (  # noqa: E402
    TORCH_DEVIATION_LINE,
    TRANSFORMERS_DEVIATION_LINE,
)


class PackageConsistencyTests(unittest.TestCase):
    """`uv pip check` reports the deviations this installer creates on purpose.

    esm is installed with --no-deps against a newer torch and Transformers than
    it declares, and its runtime requirements omit the ESMFold2-only
    cuequivariance kernels, so a clean install always reports four
    incompatibilities in uv's two shapes: an unsatisfied version and an absent
    dependency. Anything else must fail closed.
    """

    TORCH_LINE = TORCH_DEVIATION_LINE
    TRANSFORMERS_LINE = TRANSFORMERS_DEVIATION_LINE
    MISSING_LINES = tuple(
        f"The package `esm` requires `{name}>=0.8.1 ; platform_machine == "
        "'x86_64' and sys_platform == 'linux'`, but it's not installed"
        for name in sorted(Install_Dependencies.ESM_OMITTED_REQUIREMENTS)
    )

    def _check(self, *lines):
        report = "\n".join(
            ("Checked 155 packages in 1ms", f"Found {len(lines)} incompatibilities", *lines)
        )
        completed = mock.Mock(returncode=1, stdout=report, stderr="")
        stdout = io.StringIO()
        with mock.patch.object(Install_Dependencies, "_run", return_value=completed), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            return Install_Dependencies.validate_package_consistency("uv", Path("python"))

    def _expected(self):
        return (self.TORCH_LINE, self.TRANSFORMERS_LINE, *self.MISSING_LINES)

    def test_clean_report_passes(self):
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        stdout = io.StringIO()
        with mock.patch.object(Install_Dependencies, "_run", return_value=completed), \
                redirect_stdout(stdout):
            self.assertTrue(
                Install_Dependencies.validate_package_consistency("uv", Path("python"))
            )
        self.assertEqual(stdout.getvalue(), "Installed package consistency: passed\n")

    def test_sanctioned_version_and_missing_deviations_pass(self):
        self.assertTrue(self._check(*self._expected()))

    def _accepted_report(self, **options):
        report = "\n".join(("Checked 155 packages in 1ms", f"Found {len(self._expected())} incompatibilities",
                            *self._expected()))
        completed = mock.Mock(returncode=1, stdout=report, stderr="")
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(Install_Dependencies, "_run", return_value=completed), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            passed = Install_Dependencies.validate_package_consistency("uv", Path("python"), **options)
        return passed, stdout.getvalue(), stderr.getvalue()

    def test_a_readiness_check_counts_the_accepted_deviations_without_listing_them(self):
        # Every launch runs it, and a clean install always has these: the list was noise in Show details.
        passed, out, err = self._accepted_report()
        self.assertTrue(passed)
        self.assertEqual(out, f"Installed package consistency: passed with {len(self._expected())} expected esm deviation(s)\n")
        self.assertEqual(err, "")

    def test_a_setup_lists_each_accepted_deviation_once(self):
        passed, out, _ = self._accepted_report(list_accepted=True)
        self.assertTrue(passed)
        listed = [line for line in out.splitlines() if line.startswith("  accepted: ")]
        self.assertEqual(listed, [f"  accepted: {line}" for line in self._expected()])

    def test_a_failed_check_names_what_is_wrong_whether_or_not_the_list_was_asked_for(self):
        lines = (*self._expected(), "The package `esm` requires `biotite>=1.0.0`, but it's not installed")
        report = "\n".join((f"Found {len(lines)} incompatibilities", *lines))
        completed = mock.Mock(returncode=1, stdout=report, stderr="")
        for options in ({}, {"list_accepted": True}):
            stdout, stderr = io.StringIO(), io.StringIO()
            with self.subTest(options=options), mock.patch.object(Install_Dependencies, "_run", return_value=completed), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertFalse(Install_Dependencies.validate_package_consistency("uv", Path("python"), **options))
            self.assertIn("biotite>=1.0.0", stderr.getvalue())
            self.assertNotIn("accepted:", stdout.getvalue())

    def test_unsanctioned_missing_dependency_fails(self):
        lines = (
            self.TORCH_LINE,
            self.TRANSFORMERS_LINE,
            "The package `esm` requires `biotite>=1.0.0`, but it's not installed",
        )
        self.assertFalse(self._check(*lines))

    def test_missing_dependency_of_another_package_fails(self):
        lines = (
            self.TORCH_LINE,
            self.TRANSFORMERS_LINE,
            "The package `scanpy` requires `cuequivariance-torch>=0.8.1`, "
            "but it's not installed",
        )
        self.assertFalse(self._check(*lines))

    def test_unpinned_torch_or_transformers_version_fails(self):
        for line in (
            self.TORCH_LINE.replace(Install_Dependencies.TORCH_VERSION, "2.9.0"),
            self.TRANSFORMERS_LINE.replace(
                Install_Dependencies.TRANSFORMERS_VERSION, "4.57.6"
            ),
        ):
            with self.subTest(line=line):
                self.assertFalse(self._check(line))

    def test_unaccounted_or_unparseable_report_fails(self):
        # The declared count is the safety net: a line uv words differently, or
        # a report shape this installer cannot parse, must not pass silently.
        completed = mock.Mock(
            returncode=1,
            stdout="Found 5 incompatibilities\n" + "\n".join(self._expected()),
            stderr="",
        )
        with mock.patch.object(Install_Dependencies, "_run", return_value=completed), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertFalse(
                Install_Dependencies.validate_package_consistency("uv", Path("python"))
            )
        self.assertFalse(self._check("something uv started wording differently"))

    def test_omitted_requirements_are_absent_from_the_runtime_file(self):
        # The sanctioned-missing set only holds while the file really omits
        # them; listing one there would install it and silence the report.
        entries = Install_Dependencies._requirements_entries(
            Install_Dependencies._esm_runtime_requirements_path(ROOT)
        )
        names = {Install_Dependencies._requirement_name(entry) for entry in entries}
        self.assertEqual(
            names & Install_Dependencies.ESM_OMITTED_REQUIREMENTS, set()
        )


class InstallerProfileTests(unittest.TestCase):
    def test_candidates_resolving_to_the_same_install_collapse_to_one_rung(self):
        # Two AMD devices on one target used to differ by ROCm release; with a
        # single profile they resolve to an identical install, and the repeat
        # rung is dropped rather than retried.
        report = {
            "backend_candidates": [
                {"backend": "rocm", "profile": "rocm", "gfx_target": "gfx1100", "device_ids": ["amd0"]},
                {"backend": "rocm", "profile": "rocm", "gfx_target": "gfx1100", "device_ids": ["amd1"]},
                {"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]},
            ]
        }
        specs = Install_Dependencies.backend_specs(report)
        self.assertEqual([spec.backend for spec in specs], ["rocm", "cpu"])
        self.assertEqual(specs[0].device_ids, ("amd0",))

    def test_hardware_fingerprint_ignores_physical_identity_and_driver_release(self):
        report = {
            "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION,
            "platform": "linux",
            "devices": [
                {
                    "id": "0000:01:00.0", "name": "NVIDIA H100", "vendor": "NVIDIA",
                    "pci_id": "10de:2330", "driver_version": "570.1", "driver": "nvidia",
                    "kind": "discrete", "compute_capability": "9.0",
                    "eligible_profiles": ["cuda"],
                },
                {
                    "id": "0000:02:00.0", "name": "NVIDIA H100", "vendor": "NVIDIA",
                    "pci_id": "10de:2330", "driver_version": "570.1", "driver": "nvidia",
                    "kind": "discrete", "compute_capability": "9.0",
                    "eligible_profiles": ["cuda"],
                },
            ],
            "backend_candidates": [
                {"backend": "cuda126", "profile": "cuda126", "vendor": "NVIDIA", "device_ids": ["0000:01:00.0", "0000:02:00.0"]},
                {"backend": "cpu", "profile": "cpu", "vendor": "CPU", "device_ids": ["cpu"]},
            ],
        }
        first = Install_Dependencies.hardware_fingerprint(report)
        report["devices"].reverse()
        report["devices"][0]["id"] = "0000:a1:00.0"
        report["devices"][1]["id"] = "0000:a2:00.0"
        report["devices"][0]["driver_version"] = "575.9"
        report["devices"][1]["driver_version"] = "575.9"
        report["backend_candidates"][0]["device_ids"] = ["0000:a1:00.0", "0000:a2:00.0"]
        self.assertEqual(first, Install_Dependencies.hardware_fingerprint(report))

    def test_hardware_fingerprint_changes_with_compatibility_facts(self):
        report = {
            "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION,
            "platform": "linux",
            "devices": [{
                "name": "NVIDIA H100", "vendor": "NVIDIA", "pci_id": "10de:2330",
                "kind": "discrete", "compute_capability": "9.0",
                "eligible_profiles": ["cuda"],
            }],
            "backend_candidates": [
                {"backend": "cuda126", "profile": "cuda126", "vendor": "NVIDIA"},
                {"backend": "cpu", "profile": "cpu", "vendor": "CPU"},
            ],
        }
        first = Install_Dependencies.hardware_fingerprint(report)
        for mutation in ("profile", "model", "count", "revision"):
            changed = json.loads(json.dumps(report))
            if mutation == "profile":
                changed["backend_candidates"][0].update(backend="cuda132", profile="cuda132")
            elif mutation == "model":
                changed["devices"][0]["pci_id"] = "10de:2331"
            elif mutation == "count":
                changed["devices"].append(dict(changed["devices"][0]))
            else:
                changed["compatibility_revision"] += 1
            with self.subTest(mutation=mutation):
                self.assertNotEqual(first, Install_Dependencies.hardware_fingerprint(changed))

    def test_failed_cuda_falls_through_to_xpu(self):
        report = {
            "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION,
            "platform": "windows",
            "os": {"windows_build": 26200},
            "devices": [],
            "ignored_devices": [],
            "reason": "test ladder",
            "backend_candidates": [
                {"backend": "cuda126", "profile": "cuda126", "device_ids": ["n0"]},
                {"backend": "xpu", "profile": "xpu", "device_ids": ["i0"]},
                {"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]},
            ],
        }
        validation = {"validated_devices": [{"spec": "xpu:0", "success": True}]}
        completed = subprocess.CompletedProcess([], 0, "", "")
        written: dict = {}

        def capture_state(_path, payload, _report=None):
            written.update(payload)

        with tempfile.TemporaryDirectory() as folder, \
            mock.patch.object(Install_Dependencies, "venv_python", return_value=Path("python")), \
            mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report), \
            mock.patch.object(Install_Dependencies, "_run", return_value=completed), \
            mock.patch.object(Install_Dependencies, "install_backend", side_effect=[None, validation]) as install_backend, \
            mock.patch.object(Install_Dependencies, "_installed_version", return_value=None), \
            mock.patch.object(Install_Dependencies, "write_state", side_effect=capture_state):
            result = Install_Dependencies.install(
                project_root=ROOT, venv=Path(folder), uv_executable="uv"
            )
        self.assertEqual(result, 0)
        self.assertEqual([call.args[2].backend for call in install_backend.call_args_list], ["cuda126", "xpu"])
        self.assertEqual(written["active_backend"]["backend"], "xpu")

    def test_previous_compatibility_revision_is_invalidated(self):
        requirements = ROOT / "src" / "requirements.txt"
        specs = Install_Dependencies.backend_specs(
            {"backend_candidates": [{"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]}]}
        )
        current = {
            "schema": Install_Dependencies.STATE_SCHEMA,
            "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION,
            "hardware_fingerprint": "fingerprint",
            "requirements_sha256": Install_Dependencies._requirements_sha256(requirements),
            "esm_version": Install_Dependencies.ESM_VERSION,
            "transformers_version": Install_Dependencies.TRANSFORMERS_VERSION,
            "esm_runtime_requirements_sha256": Install_Dependencies._requirements_sha256(
                Install_Dependencies._esm_runtime_requirements_path(ROOT)
            ),
            "requested_candidates": Install_Dependencies._spec_payloads(specs),
        }
        self.assertTrue(
            Install_Dependencies._state_matches(current, specs, "fingerprint", requirements)
        )
        stale = dict(current)
        stale["compatibility_revision"] = Detect_GPU.COMPATIBILITY_REVISION - 1
        self.assertEqual(
            Install_Dependencies._state_mismatches(
                stale, specs, "fingerprint", requirements
            ),
            ["compatibility_revision"],
        )

    def test_dry_run_prints_every_candidate(self):
        report = {
            "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION, "platform": "windows", "os": {}, "devices": [],
            "ignored_devices": [], "reason": "test",
            "backend_candidates": [
                {"backend": "rocm", "profile": "rocm", "gfx_target": "gfx1100", "device_ids": ["a"]},
                {"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]},
            ],
        }
        with tempfile.TemporaryDirectory() as folder, \
            mock.patch.object(Install_Dependencies, "venv_python", return_value=Path("python")), \
            mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report), \
            mock.patch("builtins.print") as printer:
            result = Install_Dependencies.install(project_root=ROOT, venv=Path(folder), uv_executable="uv", dry_run=True)
        output = "\n".join(" ".join(str(value) for value in call.args) for call in printer.call_args_list)
        self.assertEqual(result, 0)
        self.assertIn("Dry run candidate 1: ROCm 10.1 (gfx1100)", output)
        self.assertIn("torch[device-gfx1100]", output)
        self.assertIn("Dry run candidate 2: CPU", output)


class BackendValidationTests(unittest.TestCase):
    """validate_backend's reading of the runtime validator's JSON report."""

    ROCM = {"backend": Install_Dependencies.ROCM_BACKEND, "gfx_target": "gfx1100"}

    def _validate(self, report, devices=(), *, package_error=None, returncode=0, stdout=None):
        spec = Install_Dependencies.backend_specs(report)[0]
        if stdout is None:
            stdout = json.dumps({
                "backend": spec.backend, "profile": spec.profile,
                "devices": list(devices), "package_error": package_error,
            })
        completed = mock.Mock(returncode=returncode, stdout=stdout, stderr="")
        with mock.patch.object(Install_Dependencies, "_run", return_value=completed), \
                redirect_stderr(io.StringIO()):
            return Install_Dependencies.validate_backend(Path("python"), spec)

    @staticmethod
    def _device(spec="cuda:0", architecture="gfx1100", success=True):
        return {"spec": spec, "architecture": architecture, "success": success, "error": None}

    def test_rocm_device_must_report_the_selected_gfx_target(self):
        # A ROCm build for another GFX target imports and even runs a tensor
        # op on some devices, so the reported architecture is the real check.
        for architecture in ("gfx1101", "gfx110", None, ""):
            with self.subTest(architecture=architecture):
                self.assertIsNone(
                    self._validate(self.ROCM, [self._device(architecture=architecture)])
                )
        for architecture in ("gfx1100", "gfx1100:sramecc+:xnack-", "GFX1100"):
            with self.subTest(architecture=architecture):
                validation = self._validate(
                    self.ROCM, [self._device(architecture=architecture)]
                )
                self.assertEqual(
                    [device["spec"] for device in validation["validated_devices"]],
                    ["cuda:0"],
                )

    def test_only_rocm_devices_on_the_selected_target_are_validated(self):
        validation = self._validate(self.ROCM, [
            self._device("cuda:0", "gfx1101"),
            self._device("cuda:1", "gfx1100:xnack-"),
            self._device("cuda:2", None),
        ])
        self.assertEqual(
            [device["spec"] for device in validation["validated_devices"]], ["cuda:1"]
        )
        mismatched, _matched, unreported = validation["devices"]
        self.assertFalse(mismatched["success"])
        self.assertIn("gfx1101 does not match selected target gfx1100", mismatched["error"])
        self.assertFalse(unreported["success"])
        self.assertIn("did not report an architecture", unreported["error"])

    def test_cuda_architecture_is_not_held_to_a_gfx_target(self):
        validation = self._validate(
            {"backend": "cuda132"}, [self._device(architecture="sm_120")]
        )
        self.assertEqual(
            [device["spec"] for device in validation["validated_devices"]], ["cuda:0"]
        )

    def test_failed_or_unreadable_reports_are_rejected(self):
        device = self._device(architecture="sm_120")
        cases = {
            "package error": dict(
                devices=[device], package_error="unexpected CUDA runtime: 12.6"
            ),
            "nonzero exit": dict(devices=[device], returncode=1),
            "non-JSON output": dict(stdout="Traceback (most recent call last):"),
            "JSON that is not an object": dict(stdout="[]"),
            "no output": dict(stdout=""),
            "no passing device": dict(
                devices=[self._device(architecture="sm_120", success=False)]
            ),
        }
        for label, arguments in cases.items():
            with self.subTest(label):
                self.assertIsNone(self._validate({"backend": "cuda132"}, **arguments))


# A stand-in `torch` package for running the generated validator programs in a
# real interpreter. fake_torch.json beside the package configures it, and
# calls.log records every device probe so a test can prove a probe never ran.
FAKE_TORCH = '''\
import json
import os
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, "fake_torch.json"), encoding="utf-8") as _handle:
    _CONFIG = json.load(_handle)


def _record(name):
    with open(os.path.join(_ROOT, "calls.log"), "a", encoding="utf-8") as handle:
        handle.write(name + "\\n")


class _Tensor:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return _Tensor(self.value + other)

    def item(self):
        return self.value


def ones(size, device=None):
    _record(f"ones:{device}")
    return _Tensor(1)


def _runtime(name, devices):
    def is_available():
        _record(f"{name}.is_available")
        return bool(devices)

    return SimpleNamespace(
        is_available=is_available,
        device_count=lambda: len(devices),
        get_device_name=lambda index: devices[index]["name"],
        get_device_properties=lambda index: SimpleNamespace(**devices[index]["properties"]),
        synchronize=lambda index=None: None,
    )


__version__ = _CONFIG["version"]
version = SimpleNamespace(cuda=_CONFIG.get("cuda"), hip=_CONFIG.get("hip"))
cuda = _runtime("cuda", _CONFIG.get("cuda_devices", []))
xpu = _runtime("xpu", _CONFIG.get("xpu_devices", []))
backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: _CONFIG.get("mps", False)))
'''


class ValidatorProgramTests(unittest.TestCase):
    """The generated validator programs, compiled and run against a fake torch.

    The programs are f-string templates with doubled braces, so a templating
    slip would surface only at install time, failing every accelerator rung.
    """

    PROFILES = (
        {"backend": "cpu"},
        {"backend": "cuda126"},
        {"backend": "cuda132"},
        {"backend": "xpu"},
        {"backend": "mps"},
        {"backend": Install_Dependencies.ROCM_BACKEND, "gfx_target": "gfx1100"},
    )
    GPU = {"name": "Test GPU", "properties": {"major": 8, "minor": 9}}

    def _validate(self, validator, report, **torch_config):
        """Run `validator` for the report's first rung against a fake torch.

        Returns the validation result, the validator's stderr and the device
        probes the program made.
        """
        spec = Install_Dependencies.backend_specs(report)[0]
        config = {"version": f"{Install_Dependencies.TORCH_VERSION}+test", **torch_config}
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder) / "torch"
            package.mkdir()
            (package / "__init__.py").write_text(FAKE_TORCH, encoding="utf-8")
            (Path(folder) / "fake_torch.json").write_text(json.dumps(config), encoding="utf-8")

            def run(command, *, capture=False):
                # -E and -S keep the caller's environment and site-packages (and
                # so the real torch) out; `-c` puts the working folder first.
                return subprocess.run(
                    [command[0], "-E", "-S", *command[1:]], cwd=folder,
                    capture_output=True, text=True, timeout=60, check=False,
                )

            errors = io.StringIO()
            with mock.patch.object(Install_Dependencies, "_run", side_effect=run), \
                    redirect_stderr(errors):
                validation = validator(Path(sys.executable), spec)
            calls = Path(folder) / "calls.log"
            probes = calls.read_text(encoding="utf-8").split() if calls.exists() else []
        return validation, errors.getvalue(), probes

    def test_generated_programs_compile_for_every_profile(self):
        for report in self.PROFILES:
            spec = Install_Dependencies.backend_specs(report)[0]
            for name, build in (
                ("runtime", Install_Dependencies._validation_program),
                ("package", Install_Dependencies._package_validation_program),
            ):
                with self.subTest(backend=spec.backend, program=name):
                    compile(build(spec), f"<{name} validator {spec.backend}>", "exec")
        compile(Install_Dependencies._esm_stack_program(), "<esm stack>", "exec")

    def test_runtime_validator_accepts_the_matching_build(self):
        rocm_gpu = {"name": "RX 7900 XTX", "properties": {"gcnArchName": "gfx1100:sramecc+:xnack-"}}
        cases = (
            ({"backend": "cpu"}, {}, [("cpu", None)]),
            (
                {"backend": "cuda126"},
                {"cuda": "12.6", "cuda_devices": [self.GPU, {"name": "B", "properties": {"major": 12, "minor": 0}}]},
                [("cuda:0", "sm_89"), ("cuda:1", "sm_120")],
            ),
            ({"backend": "cuda132"}, {"cuda": "13.2", "cuda_devices": [self.GPU]}, [("cuda:0", "sm_89")]),
            (
                {"backend": "rocm", "gfx_target": "gfx1100"},
                {"hip": "7.16", "cuda_devices": [rocm_gpu]},
                [("cuda:0", "gfx1100:sramecc+:xnack-")],
            ),
            ({"backend": "xpu"}, {"xpu_devices": [{"name": "Arc", "properties": {}}]}, [("xpu:0", None)]),
            ({"backend": "mps"}, {"mps": True}, [("mps", None)]),
        )
        for report, torch_config, expected in cases:
            with self.subTest(backend=report["backend"]):
                validation, errors, _probes = self._validate(
                    Install_Dependencies.validate_backend, report, **torch_config
                )
                self.assertIsNotNone(validation, errors)
                self.assertEqual(
                    [(item["spec"], item["architecture"]) for item in validation["validated_devices"]],
                    expected,
                )

    def test_runtime_validator_rejects_a_build_for_another_profile(self):
        cases = (
            ({"backend": "cuda132"}, {"cuda": "12.6", "cuda_devices": [self.GPU]},
             "unexpected CUDA runtime: 12.6"),
            ({"backend": "cpu"}, {"cuda": "12.6"}, "CPU profile loaded an accelerator build"),
            ({"backend": "cuda126"}, {"version": "2.11.0+cu126", "cuda": "12.6", "cuda_devices": [self.GPU]},
             "unexpected torch version: 2.11.0+cu126"),
            ({"backend": "cuda126"}, {"cuda": "12.6", "hip": "7.16", "cuda_devices": [self.GPU]},
             "CUDA profile loaded a ROCm build"),
            ({"backend": "rocm", "gfx_target": "gfx1100"}, {"cuda_devices": [self.GPU]},
             "ROCm/HIP build metadata is missing"),
        )
        for report, torch_config, message in cases:
            with self.subTest(backend=report["backend"], message=message):
                validation, errors, _probes = self._validate(
                    Install_Dependencies.validate_backend, report, **torch_config
                )
                self.assertIsNone(validation)
                self.assertIn(message, errors)

    def test_package_validator_checks_the_build_without_probing_a_device(self):
        # Control: the runtime validator's probes are recorded.
        _validation, _errors, probes = self._validate(
            Install_Dependencies.validate_backend, {"backend": "cuda126"}, cuda="12.6"
        )
        self.assertIn("cuda.is_available", probes)

        validation, errors, probes = self._validate(
            Install_Dependencies.validate_backend_package, {"backend": "cuda126"}, cuda="12.6"
        )
        self.assertIsNotNone(validation, errors)
        self.assertTrue(validation["preserved_without_accelerator"])
        self.assertEqual(probes, [])

        validation, errors, probes = self._validate(
            Install_Dependencies.validate_backend_package, {"backend": "cuda132"}, cuda="12.6"
        )
        self.assertIsNone(validation)
        self.assertIn("unexpected CUDA runtime: 12.6", errors)
        self.assertEqual(probes, [])


class LegacyExcelReaderDependencyTests(unittest.TestCase):
    """Metadata .xls uploads are read by pandas.read_excel, which needs xlrd."""

    def test_requirements_pin_xlrd_for_legacy_xls_uploads(self):
        lines = Install_Dependencies._requirement_lines(SRC / "requirements.txt")
        self.assertIn("xlrd==2.0.2", lines)

    def test_pandas_can_import_its_optional_xlrd_reader(self):
        from pandas.compat._optional import import_optional_dependency

        # Raises ImportError when xlrd is missing or older than pandas accepts.
        self.assertIsNotNone(import_optional_dependency("xlrd"))


class SimilarityThreadLimitDependencyTests(unittest.TestCase):
    """The shared similarity function limits BLAS to one thread with threadpoolctl."""

    def test_requirements_name_threadpoolctl_because_the_similarity_code_imports_it(self):
        lines = Install_Dependencies._requirement_lines(SRC / "requirements.txt")
        self.assertEqual(len([line for line in lines if line.startswith("threadpoolctl==")]), 1)
        source = (SRC / "utilities" / "Prefilter_Calibration.py").read_text(encoding="utf-8")
        self.assertIn("from threadpoolctl import threadpool_limits", source)

    def test_the_pinned_package_can_be_imported(self):
        import threadpoolctl

        self.assertTrue(callable(threadpoolctl.threadpool_limits))


class PinnedRequirementTests(unittest.TestCase):
    """Every requirement the two files list is pinned to the version a release tested."""

    FILES = ("requirements.txt", "esm_runtime_requirements.txt")
    EXACT_PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][0-9A-Za-z.+!-]*$")

    def _pins(self, name):
        return {
            Install_Dependencies._requirement_name(line): line.split("==", 1)[1]
            for line in Install_Dependencies._requirement_lines(SRC / name)
        }

    def test_every_requirement_line_pins_an_exact_version(self):
        for name in self.FILES:
            for line in Install_Dependencies._requirement_lines(SRC / name):
                with self.subTest(file=name, line=line):
                    self.assertRegex(line, self.EXACT_PIN)

    def test_a_package_in_both_files_has_the_same_pin(self):
        base, runtime = (self._pins(name) for name in self.FILES)
        shared = base.keys() & runtime.keys()
        self.assertTrue(shared)
        for package in sorted(shared):
            with self.subTest(package=package):
                self.assertEqual(base[package], runtime[package])


class SupportedHostTests(unittest.TestCase):
    """Hosts below the README's floors stop with a reason, not a uv resolution error."""

    SUPPORTED = {
        "Windows 11 x64": {"platform": "win32", "machine": "amd64", "is_64bit": True, "windows_build": 26300},
        "Windows 10 1809": {"platform": "win32", "machine": "amd64", "is_64bit": True, "windows_build": 17763},
        "Ubuntu 22.04": {"platform": "linux", "machine": "x86_64", "is_64bit": True, "glibc_version": "glibc 2.35"},
        "Debian 13": {"platform": "linux", "machine": "x86_64", "is_64bit": True, "glibc_version": "glibc 2.41"},
        "Linux, libc unknown": {"platform": "linux", "machine": "x86_64", "is_64bit": True, "glibc_version": None},
        "macOS 14 Apple Silicon": {"platform": "darwin", "machine": "arm64", "is_64bit": True, "macos_version": "14.0"},
        # An old-SDK Python reports 10.16 on any newer macOS; that is not a refusal.
        "macOS compatibility version": {"platform": "darwin", "machine": "arm64", "is_64bit": True, "macos_version": "10.16"},
    }
    UNSUPPORTED = {
        "Windows on ARM": ({"platform": "win32", "machine": "arm64", "is_64bit": True, "windows_build": 26100}, "x64"),
        "Windows 10 1803": ({"platform": "win32", "machine": "amd64", "is_64bit": True, "windows_build": 17134}, "1809"),
        "32-bit Python": ({"platform": "win32", "machine": "amd64", "is_64bit": False, "windows_build": 26300}, "64-bit"),
        "Intel Mac": ({"platform": "darwin", "machine": "x86_64", "is_64bit": True, "macos_version": "15.1"}, "Apple Silicon"),
        "macOS 13": ({"platform": "darwin", "machine": "arm64", "is_64bit": True, "macos_version": "13.6.1"}, "macOS 14"),
        "Linux ARM64": ({"platform": "linux", "machine": "aarch64", "is_64bit": True, "glibc_version": "glibc 2.39"}, "x86_64"),
        "glibc 2.31": ({"platform": "linux", "machine": "x86_64", "is_64bit": True, "glibc_version": "glibc 2.31"}, "2.34"),
        "FreeBSD": ({"platform": "freebsd14", "machine": "amd64", "is_64bit": True}, "not supported"),
    }

    def test_supported_hosts_pass(self):
        for label, facts in self.SUPPORTED.items():
            with self.subTest(label):
                self.assertIsNone(Install_Dependencies.unsupported_host_reason(facts))

    def test_unsupported_hosts_get_a_reason(self):
        for label, (facts, fragment) in self.UNSUPPORTED.items():
            with self.subTest(label):
                reason = Install_Dependencies.unsupported_host_reason(facts)
                self.assertIsNotNone(reason)
                self.assertIn(fragment, reason)

    def test_this_test_host_is_supported(self):
        facts = Install_Dependencies.host_facts()
        self.assertIsNone(Install_Dependencies.unsupported_host_reason(facts), facts)

    def _main(self, argv, *, override=None):
        facts = self.UNSUPPORTED["macOS 13"][0]
        name = Install_Dependencies.UNSUPPORTED_HOST_OVERRIDE
        environment = {key: value for key, value in os.environ.items() if key != name}
        if override:
            environment[name] = override
        stderr = io.StringIO()
        with mock.patch.object(Install_Dependencies, "host_facts", return_value=facts), \
                mock.patch.dict(os.environ, environment, clear=True), \
                mock.patch.object(Install_Dependencies.shutil, "which", return_value="/stub/uv"), \
                mock.patch.object(Install_Dependencies, "environment_is_ready", return_value=True) as ready, \
                mock.patch.object(Install_Dependencies, "install", return_value=0) as install, \
                redirect_stderr(stderr):
            code = Install_Dependencies.main(argv)
        return code, stderr.getvalue(), ready, install

    def test_setup_stops_and_says_why(self):
        code, stderr, ready, install = self._main(["--venv", "unused"])
        self.assertEqual(code, 1)
        self.assertIn("Unsupported system: EMAP-SSN needs macOS 14 or newer", stderr)
        self.assertIn(Install_Dependencies.UNSUPPORTED_HOST_OVERRIDE, stderr)
        install.assert_not_called()

    def test_readiness_check_quietly_asks_for_setup(self):
        # Every launcher answers "setup needed" by running setup, which prints the reason.
        code, stderr, ready, install = self._main(["--check-only", "--venv", "unused"])
        self.assertEqual(code, Install_Dependencies.SETUP_REQUIRED_EXIT)
        self.assertEqual(stderr, "")
        ready.assert_not_called()

    def _main_without_uv(self, argv):
        """main() on a supported host where no program named by --uv-executable exists."""
        stderr = io.StringIO()
        with mock.patch.object(Install_Dependencies, "_refuse_unsupported_host", return_value=False), \
                mock.patch.object(Install_Dependencies, "hidden_environment_files", return_value=[]), \
                mock.patch.object(Install_Dependencies.shutil, "which", return_value=None), \
                mock.patch.object(Install_Dependencies, "environment_is_ready", return_value=True) as ready, \
                mock.patch.object(Install_Dependencies, "install", return_value=0) as install, \
                redirect_stderr(stderr):
            code = Install_Dependencies.main(argv)
        return code, stderr.getvalue(), ready, install

    def test_the_check_runs_without_uv_and_keeps_its_exit_codes(self):
        for argv in (["--check-only", "--venv", "unused"],
                     ["--check-only", "--uv-executable", "no-such-uv", "--venv", "unused"],
                     ["--check-only", "--uv-executable", "", "--venv", "unused"]):
            with self.subTest(argv=argv):
                code, stderr, ready, install = self._main_without_uv(argv)
                self.assertEqual(code, 0)
                self.assertEqual(stderr, "")
                self.assertIsNone(ready.call_args.kwargs["uv_executable"])
                install.assert_not_called()
        with mock.patch.object(Install_Dependencies, "_refuse_unsupported_host", return_value=False), \
                mock.patch.object(Install_Dependencies, "hidden_environment_files", return_value=[]), \
                mock.patch.object(Install_Dependencies.shutil, "which", return_value=None), \
                mock.patch.object(Install_Dependencies, "environment_is_ready", return_value=False):
            code = Install_Dependencies.main(["--check-only", "--venv", "unused"])
        self.assertEqual(code, Install_Dependencies.SETUP_REQUIRED_EXIT)

    def test_installing_without_uv_says_so_and_changes_nothing(self):
        code, stderr, ready, install = self._main_without_uv(["--venv", "unused"])
        self.assertEqual(code, 1)
        self.assertIn("the uv package manager is not installed", stderr)
        install.assert_not_called()

    def test_override_lets_an_unsupported_host_try(self):
        code, stderr, ready, install = self._main(["--venv", "unused"], override="1")
        self.assertEqual(code, 0)
        self.assertIn("Warning:", stderr)
        install.assert_called_once()


if __name__ == "__main__":
    unittest.main()
