# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Install_Dependencies: backend specs and install commands, the saved backend
state, the install and readiness flows, the generated validator programs and
the parsing of their reports, and the `uv pip check` consistency gate.

Nothing here installs anything: uv, pip and the validators are mocked, except
that the validator programs run in a real interpreter against a fake torch.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
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
from tests.gpu_fixtures import detected_report  # noqa: E402


class DependencyInstallerTests(unittest.TestCase):
    def _cpu_report(self):
        return detected_report(
            backend_candidates=[{
                "backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]
            }],
            compatibility_revision=Detect_GPU.COMPATIBILITY_REVISION,
            platform="windows",
            os="Windows",
            devices=[],
        )

    def _cuda_report(self, backend="cuda126", device_id="0000:01:00.0"):
        return detected_report(
            backend=backend,
            backend_candidates=[
                {
                    "backend": backend, "profile": backend, "vendor": "NVIDIA",
                    "device_ids": [device_id],
                },
                {"backend": "cpu", "profile": "cpu", "vendor": "CPU", "device_ids": ["cpu"]},
            ],
            compatibility_revision=Detect_GPU.COMPATIBILITY_REVISION,
            platform="linux",
            os={"id": "rocky", "version_id": "9"},
            devices=[{
                "id": device_id, "name": "NVIDIA H100", "vendor": "NVIDIA",
                "pci_id": "10de:2330", "driver_version": "570.1", "kind": "discrete",
                "compute_capability": "9.0", "eligible_profiles": ["cuda"],
            }],
            reason="test CUDA profile",
        )

    def _rocm_report(self, gfx_target="gfx1100"):
        return detected_report(
            vendor="AMD",
            backend=Install_Dependencies.ROCM_BACKEND,
            gfx_target=gfx_target,
            reason="supported AMD test GPU",
        )

    def _pinned_versions(self):
        """Report the pinned ESM/Transformers versions as installed."""
        return lambda _python, package: (
            Install_Dependencies.TRANSFORMERS_VERSION
            if package == "transformers" else Install_Dependencies.ESM_VERSION
        )

    def test_subprocess_commands_are_not_echoed(self):
        success = mock.Mock(returncode=0, stdout="", stderr="")
        output = io.StringIO()
        with mock.patch.object(
            Install_Dependencies.subprocess, "run", return_value=success
        ) as run, redirect_stdout(output):
            completed = Install_Dependencies._run(["uv", "pip", "install", "example"])

        self.assertIs(completed, success)
        run.assert_called_once()
        self.assertEqual(output.getvalue(), "")

    def test_successful_backend_validation_is_silent(self):
        # A real validator report: an empty stdout would take the shim that
        # fabricates a payload for mocks, not the JSON path production uses.
        device = {"spec": "cpu", "success": True}
        success = mock.Mock(
            returncode=0,
            stdout=json.dumps({
                "backend": "cpu", "profile": "cpu",
                "devices": [device], "package_error": None,
            }),
            stderr="",
        )
        output = io.StringIO()
        spec = Install_Dependencies.backend_spec({"backend": "cpu"})
        with mock.patch.object(
            Install_Dependencies, "_run", return_value=success
        ), redirect_stdout(output):
            validation = Install_Dependencies.validate_backend(Path("python"), spec)

        self.assertEqual(validation["validated_devices"], [device])
        self.assertEqual(validation["validated_devices"][0]["spec"], "cpu")
        self.assertEqual(output.getvalue(), "")

    def test_package_only_validation_does_not_require_a_visible_device(self):
        success = mock.Mock(
            returncode=0,
            stdout=json.dumps({
                "backend": "cuda126", "profile": "cuda126",
                "torch_version": f"{Install_Dependencies.TORCH_VERSION}+cu126",
                "package_error": None,
            }),
            stderr="",
        )
        spec = Install_Dependencies.backend_spec({"backend": "cuda126"})
        with mock.patch.object(Install_Dependencies, "_run", return_value=success) as run:
            validation = Install_Dependencies.validate_backend_package(
                Path("python"), spec
            )
        self.assertEqual(validation["validated_devices"], [])
        self.assertTrue(validation["preserved_without_accelerator"])
        program = run.call_args.args[0][-1]
        self.assertNotIn("torch.cuda.is_available", program)
        self.assertNotIn("torch.ones", program)

    def test_esm_import_smoke_test_forces_utf8_mode(self):
        success = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(
            Install_Dependencies, "_run", return_value=success
        ) as run:
            self.assertTrue(Install_Dependencies.validate_esm_stack(Path("python")))
        command = run.call_args.args[0]
        self.assertEqual(command[1:3], ["-X", "utf8"])

    def test_backend_commands_use_exact_versions_and_indexes(self):
        python = Path(".venv/Scripts/python.exe")
        index_suffixes = {
            "cpu": "/cpu",
            "cuda126": "/cu126",
            "cuda132": "/cu132",
            "xpu": "/xpu",
        }
        for backend, index_suffix in index_suffixes.items():
            with self.subTest(backend=backend):
                spec = Install_Dependencies.backend_spec({"backend": backend})
                command = Install_Dependencies.torch_install_command("uv", python, spec)
                self.assertIn(f"torch=={Install_Dependencies.TORCH_VERSION}", command)
                self.assertTrue(command[-1].endswith(index_suffix))

        mps = Install_Dependencies.backend_spec({"backend": "mps"})
        self.assertNotIn("--index-url", Install_Dependencies.torch_install_command("uv", python, mps))

    def test_rocm_is_one_architecture_specific_profile(self):
        # The per-ROCm-release profiles are gone: a single `rocm` backend
        # installs from AMD's multi-arch channel and selects the device
        # package with the validated GFX target.
        self.assertEqual(
            sorted(
                backend for backend in Install_Dependencies.ACCELERATOR_BACKENDS
                if backend.startswith("rocm")
            ),
            [Install_Dependencies.ROCM_BACKEND],
        )
        spec = Install_Dependencies.backend_spec(
            {"backend": Install_Dependencies.ROCM_BACKEND, "gfx_target": "gfx1100"}
        )
        command = Install_Dependencies.torch_install_command("uv", Path("python"), spec)
        self.assertIn(
            f"torch[device-gfx1100]=={Install_Dependencies.ROCM_TORCH_VERSION}", command
        )
        self.assertEqual(command[-1], "https://repo.amd.com/rocm/whl-multi-arch/")
        # One step on every platform: torch with its device extra is the whole install.
        self.assertEqual(len(spec.install_steps), 1)
        # The recorded version drops the +rocm local segment so it stays
        # comparable with torch.__version__ during runtime validation.
        self.assertEqual(spec.torch_version, Install_Dependencies.TORCH_VERSION)
        with self.assertRaises(ValueError):
            Install_Dependencies.backend_spec(
                {"backend": Install_Dependencies.ROCM_BACKEND}
            )

    def test_malformed_local_state_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            path.write_text("{broken", encoding="utf-8")
            self.assertIsNone(Install_Dependencies.read_state(path))

    def test_backend_cleanup_includes_xpu_and_rocm_runtime_packages(self):
        for name in (
            "torch",
            "triton-xpu",
            "intel-opencl-rt",
            "onemkl-sycl-blas",
            "amd-torch-device-gfx1100",
            "rocm-sdk-core",
        ):
            with self.subTest(name=name):
                self.assertTrue(Install_Dependencies._is_backend_package(name))

    def test_esm_runtime_requirements_never_pin_torch_or_transformers(self):
        # esm is installed with --no-deps, so this file is the only place its
        # runtime dependencies are declared. torch comes from the accelerator
        # index and transformers is pinned separately; either one listed here
        # would silently replace the selected build.
        path = Install_Dependencies._esm_runtime_requirements_path(ROOT)
        Install_Dependencies.verify_esm_runtime_requirements(path)
        names = {
            Install_Dependencies._requirement_name(entry)
            for entry in Install_Dependencies._requirements_entries(path)
        }
        self.assertEqual(names & {"torch", "transformers"}, set())

        with tempfile.TemporaryDirectory() as temp_dir:
            rejected = Path(temp_dir) / "rejected.txt"
            for line in ("torch==2.12.0", "Transformers >= 5.17.0", "torch"):
                with self.subTest(line=line):
                    rejected.write_text(f"# comment\neinops\n{line}\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "must not pin"):
                        Install_Dependencies.verify_esm_runtime_requirements(rejected)

            # A distribution whose name merely starts with "torch" is a
            # different package and must not be rejected.
            accepted = Path(temp_dir) / "accepted.txt"
            accepted.write_text("# comment\ntorch_geometric\neinops\n", encoding="utf-8")
            Install_Dependencies.verify_esm_runtime_requirements(accepted)
            self.assertEqual(
                Install_Dependencies._requirements_entries(accepted),
                ("torch_geometric", "einops"),
            )

            with self.assertRaises(FileNotFoundError):
                Install_Dependencies.verify_esm_runtime_requirements(
                    Path(temp_dir) / "missing.txt"
                )

    def test_runtime_requirements_are_declared_for_the_pinned_esm_version(self):
        # esm no longer ships as a bundled wheel, so its Requires-Dist metadata
        # cannot be read back here. The file records the release it was derived
        # from instead, and the installer requires it to be rewritten whenever
        # ESM_VERSION moves; this keeps the two from drifting apart silently.
        header = Install_Dependencies._esm_runtime_requirements_path(ROOT).read_text(
            encoding="utf-8"
        ).splitlines()[0]
        self.assertEqual(
            header,
            f"# ESM runtime dependencies (esm {Install_Dependencies.ESM_VERSION})",
        )

    def test_install_order_uses_transformers_dependencies_then_esm_no_deps(self):
        report = self._cpu_report()
        success = mock.Mock(returncode=0, stdout="", stderr="")
        commands = []
        events = []

        def run(command, **_kwargs):
            commands.append(command)
            events.append(" ".join(str(part) for part in command))
            return success

        def install_backend(*_args):
            events.append("BACKEND")
            return True

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Install_Dependencies, "venv_python", return_value=Path(sys.executable)
        ), mock.patch.object(
            Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
        ), mock.patch.object(
            Install_Dependencies, "_run", side_effect=run
        ), mock.patch.object(
            Install_Dependencies, "install_backend", side_effect=install_backend
        ), mock.patch.object(
            Install_Dependencies, "_installed_version", return_value=None
        ), mock.patch.object(Install_Dependencies, "write_state"):
            code = Install_Dependencies.install(
                project_root=ROOT, venv=Path(temp_dir), uv_executable="uv"
            )

        self.assertEqual(code, 0)
        transformers_requirement = (
            f"transformers=={Install_Dependencies.TRANSFORMERS_VERSION}"
        )
        esm_requirement = f"esm=={Install_Dependencies.ESM_VERSION}"
        runtime_requirements = str(ROOT / "src" / "esm_runtime_requirements.txt")
        positions = {
            "base": next(i for i, value in enumerate(events) if str(ROOT / "src" / "requirements.txt") in value),
            "backend": events.index("BACKEND"),
            "transformers": next(i for i, value in enumerate(events) if transformers_requirement in value),
            "runtime": next(i for i, value in enumerate(events) if runtime_requirements in value),
            "esm": next(i for i, value in enumerate(events) if esm_requirement in value),
            "check": next(i for i, value in enumerate(events) if "pip check" in value),
        }
        self.assertEqual(list(positions.values()), sorted(positions.values()))
        transformers_command = next(
            command for command in commands
            if transformers_requirement in " ".join(str(part) for part in command)
        )
        esm_command = next(
            command for command in commands
            if esm_requirement in " ".join(str(part) for part in command)
        )
        # Transformers resolves its own dependencies; esm must not, because it
        # pins an older torch and Transformers than this installer selects.
        self.assertNotIn("--no-deps", transformers_command)
        self.assertIn("--no-deps", esm_command)
        self.assertLess(esm_command.index("--no-deps"), len(esm_command) - 1)
        # The hand-maintained runtime requirements replace what --no-deps skips.
        runtime_command = next(
            command for command in commands
            if runtime_requirements in " ".join(str(part) for part in command)
        )
        self.assertEqual(runtime_command[-2:], ["-r", runtime_requirements])

    def test_dry_run_names_pypi_packages_and_the_runtime_requirements_file(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Install_Dependencies, "venv_python", return_value=Path(sys.executable)
        ), mock.patch.object(
            Install_Dependencies.Detect_GPU,
            "detect_hardware",
            return_value=self._cpu_report(),
        ), redirect_stdout(output):
            code = Install_Dependencies.install(
                project_root=ROOT,
                venv=Path(temp_dir),
                uv_executable="uv",
                dry_run=True,
            )
        text = output.getvalue()
        self.assertEqual(code, 0)
        self.assertIn(f"transformers=={Install_Dependencies.TRANSFORMERS_VERSION}", text)
        self.assertIn(f"esm=={Install_Dependencies.ESM_VERSION}", text)
        self.assertIn(str(ROOT / "src" / "esm_runtime_requirements.txt"), text)
        self.assertIn("--no-deps", text)
        self.assertIn(f"torch=={Install_Dependencies.TORCH_VERSION}", text)
        # Both packages come from PyPI now: no bundled wheel and no fork.
        self.assertNotIn(".whl", text)
        self.assertNotIn("github.com", text)

    def test_unusable_runtime_requirements_fail_before_any_install_command(self):
        errors = (
            FileNotFoundError("ESM runtime requirements are missing"),
            ValueError("ESM runtime requirements must not pin torch"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temp_dir, \
                    mock.patch.object(Install_Dependencies, "venv_python", return_value=Path(sys.executable)), \
                    mock.patch.object(
                        Install_Dependencies,
                        "verify_esm_runtime_requirements",
                        side_effect=error,
                    ), \
                    mock.patch.object(Install_Dependencies, "_run") as run:
                with self.assertRaises(type(error)):
                    Install_Dependencies.install(
                        project_root=ROOT, venv=Path(temp_dir), uv_executable="uv"
                    )
                run.assert_not_called()

    def test_wrong_installed_pin_makes_the_environment_not_ready(self):
        report = self._cpu_report()
        spec = Install_Dependencies.backend_specs(report)[0]
        stale = {"transformers": "4.57.6", "esm": "3.3.0"}
        for package, version in stale.items():
            with self.subTest(package=package), tempfile.TemporaryDirectory() as temp_dir:
                venv = Path(temp_dir)
                Install_Dependencies.write_state(
                    venv / Install_Dependencies.STATE_FILENAME,
                    Install_Dependencies._state_profile(
                        spec, ROOT / "src" / "requirements.txt"
                    ),
                    report,
                )
                pinned = self._pinned_versions()
                with mock.patch.object(
                    Install_Dependencies, "venv_python", return_value=Path(sys.executable)
                ), mock.patch.object(
                    Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
                ), mock.patch.object(
                    Install_Dependencies,
                    "validate_backend",
                    return_value={"validated_devices": [{"spec": "cpu", "success": True}]},
                ), mock.patch.object(
                    Install_Dependencies,
                    "_installed_version",
                    side_effect=lambda python, name: (
                        version if name == package else pinned(python, name)
                    ),
                ), mock.patch.object(
                    Install_Dependencies, "validate_package_consistency", return_value=True
                ), mock.patch.object(
                    Install_Dependencies, "validate_esm_stack", return_value=True
                ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    ready = Install_Dependencies.environment_is_ready(
                        project_root=ROOT, venv=venv, uv_executable="uv"
                    )
                self.assertFalse(ready)

    def test_state_schema_versions_hashes_and_runtime_requirements_invalidate(self):
        report = self._cpu_report()
        specs = Install_Dependencies.backend_specs(report)
        requirements = ROOT / "src" / "requirements.txt"
        state = Install_Dependencies._state_profile(specs[0], requirements)
        state.update({
            "hardware_fingerprint": Install_Dependencies.hardware_fingerprint(report),
            "requested_candidates": Install_Dependencies._spec_payloads(specs),
        })
        fingerprint = Install_Dependencies.hardware_fingerprint(report)
        self.assertTrue(
            Install_Dependencies._state_matches(state, specs, fingerprint, requirements)
        )
        mutations = {
            "schema": Install_Dependencies.STATE_SCHEMA - 1,
            "esm_version": "3.3.0",
            "transformers_version": "4.57.6",
            "requirements_sha256": "0" * 64,
            "esm_runtime_requirements_sha256": "0" * 64,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = dict(state)
                changed[field] = value
                self.assertFalse(
                    Install_Dependencies._state_matches(
                        changed, specs, fingerprint, requirements
                    )
                )

    def test_state_mismatches_name_every_changed_field_in_order(self):
        report = self._cpu_report()
        specs = Install_Dependencies.backend_specs(report)
        requirements = ROOT / "src" / "requirements.txt"
        fingerprint = Install_Dependencies.hardware_fingerprint(report)
        state = Install_Dependencies._state_profile(specs[0], requirements)
        state.update({
            "hardware_fingerprint": "old-fingerprint",
            "requested_candidates": [],
            "schema": Install_Dependencies.STATE_SCHEMA - 1,
            "requirements_sha256": "old-requirements",
            "esm_version": "old-esm",
        })
        self.assertEqual(
            Install_Dependencies._state_mismatches(
                state, specs, fingerprint, requirements
            ),
            [
                "schema", "hardware_fingerprint", "requirements_sha256",
                "esm_version", "requested_candidates",
            ],
        )

    def _requirement_fingerprint(self, data):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "requirements.txt"
            path.write_bytes(data)
            return Install_Dependencies._requirements_sha256(path)

    def test_requirement_fingerprint_ignores_comments_blank_lines_and_line_endings(self):
        base = self._requirement_fingerprint(b"# Core\nnumpy==2.5.3\nscipy==1.18.1\n")
        for variant in (
            b"# Core, reworded\nnumpy==2.5.3\n\n# More\nscipy==1.18.1\n",
            b"numpy==2.5.3  # numba ceiling\n  scipy==1.18.1\t\n",
            b"# Core\r\nnumpy==2.5.3\r\nscipy==1.18.1\r\n",
            b"\xef\xbb\xbf# Core\nnumpy==2.5.3\nscipy==1.18.1",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(self._requirement_fingerprint(variant), base)

    def test_requirement_fingerprint_changes_with_anything_pip_reads(self):
        base = self._requirement_fingerprint(b"numpy==2.5.3\nscipy==1.18.1\n")
        for variant in (
            b"numpy==2.5.4\nscipy==1.18.1\n",
            b"numpy==2.5.3\nscipy==1.18.1\npandas==3.0.6\n",
            b"numpy==2.5.3\n",
            b"scipy==1.18.1\nnumpy==2.5.3\n",
            b"numpy==2.5.3 ; sys_platform == 'linux'\nscipy==1.18.1\n",
            b"--index-url https://example.org/simple\nnumpy==2.5.3\nscipy==1.18.1\n",
        ):
            with self.subTest(variant=variant):
                self.assertNotEqual(self._requirement_fingerprint(variant), base)

    def test_a_url_fragment_is_requirement_content_not_a_comment(self):
        # pip treats "#" as a comment only at a line start or after whitespace.
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "requirements.txt"
            path.write_text(
                "pkg @ git+https://example.org/pkg.git#egg=pkg  # pinned fork\n",
                encoding="utf-8",
            )
            self.assertEqual(
                Install_Dependencies._requirement_lines(path),
                ("pkg @ git+https://example.org/pkg.git#egg=pkg",),
            )

    def test_a_comment_or_line_ending_edit_keeps_the_saved_state_current(self):
        report = self._cpu_report()
        specs = Install_Dependencies.backend_specs(report)
        fingerprint = Install_Dependencies.hardware_fingerprint(report)
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements = Path(temp_dir) / "requirements.txt"
            runtime = Path(temp_dir) / "esm_runtime_requirements.txt"
            requirements.write_text("# Core\nnumpy==2.5.3\n", encoding="utf-8")
            runtime.write_text("# ESM runtime\neinops\n", encoding="utf-8")
            state = Install_Dependencies._state_profile(specs[0], requirements)
            state.update({
                "hardware_fingerprint": fingerprint,
                "requested_candidates": Install_Dependencies._spec_payloads(specs),
            })

            requirements.write_bytes(
                b"# Core, reworded\r\nnumpy==2.5.3  # numba ceiling\r\n\r\n"
            )
            runtime.write_text("# ESM runtime, reworded\n\neinops\n", encoding="utf-8")
            self.assertEqual(
                Install_Dependencies._state_mismatches(
                    state, specs, fingerprint, requirements
                ),
                [],
            )

            requirements.write_text("numpy==2.5.4\n", encoding="utf-8")
            self.assertEqual(
                Install_Dependencies._state_mismatches(
                    state, specs, fingerprint, requirements
                ),
                ["requirements_sha256"],
            )

    def test_stale_metadata_reuses_compatible_cuda_without_reinstall(self):
        report = self._cuda_report()
        specs = Install_Dependencies.backend_specs(report)
        requirements = ROOT / "src" / "requirements.txt"
        success = mock.Mock(returncode=0, stdout="", stderr="")
        validation = {"validated_devices": [{"spec": "cuda:0", "success": True}]}
        with tempfile.TemporaryDirectory() as temp_dir:
            venv = Path(temp_dir)
            state = Install_Dependencies._state_profile(specs[0], requirements)
            Install_Dependencies.write_state(
                venv / Install_Dependencies.STATE_FILENAME, state, report
            )
            saved = Install_Dependencies.read_state(
                venv / Install_Dependencies.STATE_FILENAME
            )
            saved["schema"] = Install_Dependencies.STATE_SCHEMA - 1
            (venv / Install_Dependencies.STATE_FILENAME).write_text(
                json.dumps(saved), encoding="utf-8"
            )
            with mock.patch.object(
                Install_Dependencies, "venv_python", return_value=Path(sys.executable)
            ), mock.patch.object(
                Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
            ), mock.patch.object(
                Install_Dependencies, "_run", return_value=success
            ), mock.patch.object(
                Install_Dependencies, "validate_backend", return_value=validation
            ) as validate, mock.patch.object(
                Install_Dependencies, "install_backend"
            ) as install_backend, mock.patch.object(
                Install_Dependencies, "_installed_version",
                side_effect=self._pinned_versions(),
            ), mock.patch.object(
                Install_Dependencies, "validate_package_consistency", return_value=True
            ), mock.patch.object(
                Install_Dependencies, "validate_esm_stack", return_value=True
            ), mock.patch.object(Install_Dependencies, "write_state"):
                code = Install_Dependencies.install(
                    project_root=ROOT, venv=venv, uv_executable="uv"
                )

        self.assertEqual(code, 0)
        self.assertEqual(validate.call_args.args[1].backend, "cuda126")
        install_backend.assert_not_called()

    def test_cpu_only_node_preserves_installed_accelerator_package(self):
        cuda_report = self._cuda_report()
        cpu_report = self._cpu_report()
        cuda_spec = Install_Dependencies.backend_specs(cuda_report)[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            venv = Path(temp_dir)
            python = venv / "python"
            python.touch()
            Install_Dependencies.write_state(
                venv / Install_Dependencies.STATE_FILENAME,
                Install_Dependencies._state_profile(
                    cuda_spec, ROOT / "src" / "requirements.txt"
                ),
                cuda_report,
            )
            with mock.patch.object(
                Install_Dependencies, "venv_python", return_value=python
            ), mock.patch.object(
                Install_Dependencies.Detect_GPU,
                "detect_hardware",
                return_value=cpu_report,
            ), mock.patch.object(
                Install_Dependencies,
                "validate_backend_package",
                return_value={"validated_devices": []},
            ) as package_validation, mock.patch.object(
                Install_Dependencies, "validate_backend"
            ) as runtime_validation, mock.patch.object(
                Install_Dependencies, "_installed_version",
                side_effect=self._pinned_versions(),
            ), mock.patch.object(
                Install_Dependencies, "validate_package_consistency", return_value=True
            ), mock.patch.object(
                Install_Dependencies, "validate_esm_stack", return_value=True
            ):
                ready = Install_Dependencies.environment_is_ready(
                    project_root=ROOT, venv=venv, uv_executable="uv"
                )

        self.assertTrue(ready)
        package_validation.assert_called_once()
        runtime_validation.assert_not_called()

    def test_refresh_backend_on_cpu_only_node_explicitly_installs_cpu(self):
        cuda_report = self._cuda_report()
        cpu_report = self._cpu_report()
        cuda_spec = Install_Dependencies.backend_specs(cuda_report)[0]
        success = mock.Mock(returncode=0, stdout="", stderr="")
        validation = {"validated_devices": [{"spec": "cpu", "success": True}]}
        with tempfile.TemporaryDirectory() as temp_dir:
            venv = Path(temp_dir)
            Install_Dependencies.write_state(
                venv / Install_Dependencies.STATE_FILENAME,
                Install_Dependencies._state_profile(
                    cuda_spec, ROOT / "src" / "requirements.txt"
                ),
                cuda_report,
            )
            with mock.patch.object(
                Install_Dependencies, "venv_python", return_value=Path(sys.executable)
            ), mock.patch.object(
                Install_Dependencies.Detect_GPU, "detect_hardware", return_value=cpu_report
            ), mock.patch.object(
                Install_Dependencies, "_run", return_value=success
            ), mock.patch.object(
                Install_Dependencies, "install_backend", return_value=validation
            ) as install_backend, mock.patch.object(
                Install_Dependencies, "validate_backend_package"
            ) as package_validation, mock.patch.object(
                Install_Dependencies, "_installed_version",
                side_effect=self._pinned_versions(),
            ), mock.patch.object(
                Install_Dependencies, "validate_package_consistency", return_value=True
            ), mock.patch.object(
                Install_Dependencies, "validate_esm_stack", return_value=True
            ), mock.patch.object(Install_Dependencies, "write_state"):
                code = Install_Dependencies.install(
                    project_root=ROOT, venv=venv, uv_executable="uv",
                    refresh_backend=True,
                )

        self.assertEqual(code, 0)
        self.assertEqual(install_backend.call_args.args[2].backend, "cpu")
        package_validation.assert_not_called()

    def test_changed_backend_profile_and_new_accelerator_are_not_reused(self):
        cuda126_specs = Install_Dependencies.backend_specs(self._cuda_report("cuda126"))
        cuda132_specs = Install_Dependencies.backend_specs(self._cuda_report("cuda132"))
        cpu_specs = Install_Dependencies.backend_specs(self._cpu_report())
        cuda_state = {
            "active_backend": Install_Dependencies._spec_payloads((cuda126_specs[0],))[0],
            "requested_candidates": Install_Dependencies._spec_payloads(cuda126_specs),
        }
        cpu_state = {
            "active_backend": Install_Dependencies._spec_payloads((cpu_specs[0],))[0],
            "requested_candidates": Install_Dependencies._spec_payloads(cpu_specs),
        }
        self.assertIsNone(
            Install_Dependencies._reusable_backend(cuda_state, cuda132_specs)
        )
        self.assertIsNone(
            Install_Dependencies._reusable_backend(cpu_state, cuda126_specs)
        )

    def test_provisional_accelerator_inventory_preserves_saved_build(self):
        saved_specs = Install_Dependencies.backend_specs(self._cuda_report("cuda132"))
        provisional_report = self._cuda_report("cuda126")
        provisional_report["backend_candidates"][0]["eligibility"] = "provisional"
        current_specs = Install_Dependencies.backend_specs(provisional_report)
        state = {
            "active_backend": Install_Dependencies._spec_payloads((saved_specs[0],))[0],
            "requested_candidates": Install_Dependencies._spec_payloads(saved_specs),
        }
        self.assertFalse(
            Install_Dependencies._accelerator_runtime_visible(provisional_report)
        )
        reusable = Install_Dependencies._reusable_backend(
            state, current_specs, accelerator_visible=False
        )
        self.assertIsNotNone(reusable)
        self.assertEqual(reusable[0].backend, "cuda132")
        self.assertEqual(reusable[1], "package-only")

    def test_provisional_gpu_keeps_cpu_but_eligible_gpu_upgrades_it(self):
        cpu_report = self._cpu_report()
        cpu_spec = Install_Dependencies.backend_specs(cpu_report)[0]
        provisional_report = self._cuda_report("cuda126")
        provisional_report["backend_candidates"][0]["eligibility"] = "provisional"
        provisional_specs = Install_Dependencies.backend_specs(provisional_report)
        cpu_state = {
            "active_backend": Install_Dependencies._spec_payloads((cpu_spec,))[0],
            "requested_candidates": Install_Dependencies._spec_payloads((cpu_spec,)),
            "detection": cpu_report,
        }
        reusable = Install_Dependencies._reusable_backend(
            cpu_state, provisional_specs, accelerator_visible=False
        )
        self.assertEqual(reusable[0].backend, "cpu")

        provisional_state = {
            "active_backend": Install_Dependencies._spec_payloads((cpu_spec,))[0],
            "requested_candidates": Install_Dependencies._spec_payloads(provisional_specs),
            "detection": provisional_report,
        }
        self.assertIsNone(
            Install_Dependencies._reusable_backend(
                provisional_state, provisional_specs, accelerator_visible=True
            )
        )

    def test_failed_visible_runtime_validation_reinstalls_same_backend(self):
        report = self._cuda_report()
        specs = Install_Dependencies.backend_specs(report)
        success = mock.Mock(returncode=0, stdout="", stderr="")
        repaired = {"validated_devices": [{"spec": "cuda:0", "success": True}]}
        with tempfile.TemporaryDirectory() as temp_dir:
            venv = Path(temp_dir)
            Install_Dependencies.write_state(
                venv / Install_Dependencies.STATE_FILENAME,
                Install_Dependencies._state_profile(
                    specs[0], ROOT / "src" / "requirements.txt"
                ),
                report,
            )
            with mock.patch.object(
                Install_Dependencies, "venv_python", return_value=Path(sys.executable)
            ), mock.patch.object(
                Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
            ), mock.patch.object(
                Install_Dependencies, "_run", return_value=success
            ), mock.patch.object(
                Install_Dependencies, "validate_backend", return_value=None
            ), mock.patch.object(
                Install_Dependencies, "install_backend", return_value=repaired
            ) as install_backend, mock.patch.object(
                Install_Dependencies, "_installed_version",
                side_effect=self._pinned_versions(),
            ), mock.patch.object(
                Install_Dependencies, "validate_package_consistency", return_value=True
            ), mock.patch.object(
                Install_Dependencies, "validate_esm_stack", return_value=True
            ), mock.patch.object(Install_Dependencies, "write_state"):
                code = Install_Dependencies.install(
                    project_root=ROOT, venv=venv, uv_executable="uv"
                )

        self.assertEqual(code, 0)
        self.assertEqual(install_backend.call_args.args[2].backend, "cuda126")

    def test_failed_accelerator_install_falls_back_to_cpu(self):
        report = self._rocm_report()
        success = mock.Mock(returncode=0, stdout="", stderr="")
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(Install_Dependencies, "venv_python", return_value=Path(sys.executable)), \
                mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report), \
                mock.patch.object(Install_Dependencies, "_run", return_value=success), \
                mock.patch.object(Install_Dependencies, "install_backend", side_effect=[False, True]) as install_backend, \
                mock.patch.object(Install_Dependencies, "_installed_version", return_value=None), \
                mock.patch.object(Install_Dependencies, "write_state") as write_state, \
                redirect_stdout(output):
            code = Install_Dependencies.install(
                project_root=ROOT,
                venv=Path(temp_dir),
                uv_executable="uv",
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            install_backend.call_args_list[0].args[2].backend,
            Install_Dependencies.ROCM_BACKEND,
        )
        self.assertEqual(install_backend.call_args_list[1].args[2].backend, "cpu")
        self.assertEqual(
            write_state.call_args.args[2]["fallback_from"],
            Install_Dependencies.ROCM_BACKEND,
        )
        # The saved state keeps the requested ladder apart from the active build.
        saved = write_state.call_args.args[1]
        self.assertEqual(saved["active_backend"]["backend"], "cpu")
        self.assertEqual(
            [candidate["backend"] for candidate in saved["requested_candidates"]],
            [Install_Dependencies.ROCM_BACKEND, "cpu"],
        )
        self.assertEqual(saved["requested_candidates"][0]["gfx_target"], "gfx1100")
        status = output.getvalue()
        self.assertNotIn("$ ", status)
        self.assertNotIn("Selected PyTorch backend:", status)
        self.assertNotIn("Validated PyTorch backend:", status)
        self.assertIn("Dependency environment is ready (CPU).", status)

    def test_validated_cpu_fallback_is_reused_for_the_same_hardware(self):
        report = self._rocm_report()
        requested = Install_Dependencies.backend_spec(report)
        active = Install_Dependencies.backend_spec({"backend": "cpu"})
        success = mock.Mock(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as temp_dir:
            venv = Path(temp_dir)
            Install_Dependencies.write_state(
                venv / Install_Dependencies.STATE_FILENAME,
                Install_Dependencies._state_profile(
                    active, ROOT / "src" / "requirements.txt", requested
                ),
                report,
            )
            with mock.patch.object(
                    Install_Dependencies, "venv_python", return_value=Path(sys.executable)
                ), mock.patch.object(
                    Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
                ), mock.patch.object(
                    Install_Dependencies, "_run", return_value=success
                ), mock.patch.object(
                    Install_Dependencies, "validate_backend", return_value=True
                ) as validate, mock.patch.object(
                    Install_Dependencies, "install_backend"
                ) as install_backend, mock.patch.object(
                    Install_Dependencies, "_installed_version",
                    side_effect=self._pinned_versions(),
                ):
                code = Install_Dependencies.install(
                    project_root=ROOT,
                    venv=venv,
                    uv_executable="uv",
                )
        self.assertEqual(code, 0)
        self.assertEqual(validate.call_args.args[1].backend, "cpu")
        install_backend.assert_not_called()


class PackageConsistencyTests(unittest.TestCase):
    """`uv pip check` reports the deviations this installer creates on purpose.

    esm is installed with --no-deps against a newer torch and Transformers than
    it declares, and its runtime requirements omit the ESMFold2-only
    cuequivariance kernels, so a clean install always reports four
    incompatibilities in uv's two shapes: an unsatisfied version and an absent
    dependency. Anything else must fail closed.
    """

    TORCH_LINE = (
        "The package `esm` requires `torch>=2.11.0,<2.12.0`, "
        f"but `{Install_Dependencies.TORCH_VERSION}+cu132` is installed"
    )
    TRANSFORMERS_LINE = (
        "The package `esm` requires `transformers>=4.57.6,<5.0.0`, "
        f"but `{Install_Dependencies.TRANSFORMERS_VERSION}` is installed"
    )
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
        with mock.patch.object(Install_Dependencies, "_run", return_value=completed):
            self.assertTrue(
                Install_Dependencies.validate_package_consistency("uv", Path("python"))
            )

    def test_sanctioned_version_and_missing_deviations_pass(self):
        self.assertTrue(self._check(*self._expected()))

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


class DependencyReadinessTests(unittest.TestCase):
    def test_ready_environment_is_checked_without_installing(self):
        completed = mock.Mock(returncode=0)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python = root / "python"
            python.touch()
            state = {"active_backend": {"backend": "cpu"}}
            active = Install_Dependencies.backend_spec({"backend": "cpu"})
            with mock.patch.object(Install_Dependencies, "venv_python", return_value=python), \
                    mock.patch.object(Install_Dependencies, "verify_esm_runtime_requirements"), \
                    mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value={}), \
                    mock.patch.object(Install_Dependencies, "backend_specs", return_value=[]), \
                    mock.patch.object(Install_Dependencies, "hardware_fingerprint", return_value="fp"), \
                    mock.patch.object(Install_Dependencies, "read_state", return_value=state), \
                    mock.patch.object(Install_Dependencies, "_state_mismatches", return_value=[]), \
                    mock.patch.object(Install_Dependencies, "_backend_from_state", return_value=active), \
                    mock.patch.object(Install_Dependencies, "validate_backend", return_value={"devices": []}), \
                    mock.patch.object(
                        Install_Dependencies,
                        "_installed_version",
                        side_effect=lambda _python, package: (
                            Install_Dependencies.TRANSFORMERS_VERSION
                            if package == "transformers"
                            else Install_Dependencies.ESM_VERSION
                        ),
                    ), mock.patch.object(
                        Install_Dependencies, "validate_package_consistency", return_value=True
                    ) as consistency, mock.patch.object(
                        Install_Dependencies, "validate_esm_stack", return_value=True
                    ):
                ready = Install_Dependencies.environment_is_ready(
                    project_root=root, venv=root, uv_executable="uv"
                )

        self.assertTrue(ready)
        consistency.assert_called_once_with("uv", python)

    def test_changed_state_requires_setup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python = root / "python"
            python.touch()
            with mock.patch.object(Install_Dependencies, "venv_python", return_value=python), \
                    mock.patch.object(Install_Dependencies, "verify_esm_runtime_requirements"), \
                    mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value={}), \
                    mock.patch.object(Install_Dependencies, "backend_specs", return_value=[]), \
                    mock.patch.object(Install_Dependencies, "hardware_fingerprint", return_value="fp"), \
                    mock.patch.object(Install_Dependencies, "read_state", return_value={}), \
                    mock.patch.object(
                        Install_Dependencies,
                        "_state_mismatches",
                        return_value=["hardware_fingerprint"],
                    ), \
                    mock.patch.object(Install_Dependencies, "validate_backend") as validate:
                ready = Install_Dependencies.environment_is_ready(
                    project_root=root, venv=root, uv_executable="uv"
                )

        self.assertFalse(ready)
        validate.assert_not_called()


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
            "compatibility_revision": 3,
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
            "compatibility_revision": 3, "platform": "windows", "os": {}, "devices": [],
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
        self.assertIn("Dry run candidate 1: ROCm 7.14 (gfx1100)", output)
        self.assertIn("torch[device-gfx1100]", output)
        self.assertIn("Dry run candidate 2: CPU", output)


class BackendValidationTests(unittest.TestCase):
    """validate_backend's reading of the runtime validator's JSON report."""

    ROCM = {"backend": Install_Dependencies.ROCM_BACKEND, "gfx_target": "gfx1100"}

    def _validate(self, report, devices=(), *, package_error=None, returncode=0, stdout=None):
        spec = Install_Dependencies.backend_spec(report)
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
        spec = Install_Dependencies.backend_spec(report)
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
            spec = Install_Dependencies.backend_spec(report)
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
                {"hip": "7.14", "cuda_devices": [rocm_gpu]},
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
            ({"backend": "cuda126"}, {"cuda": "12.6", "hip": "7.14", "cuda_devices": [self.GPU]},
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


if __name__ == "__main__":
    unittest.main()
