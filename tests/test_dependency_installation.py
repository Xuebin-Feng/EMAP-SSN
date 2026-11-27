# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Install_Dependencies: install and readiness flows, saved backend state, and
hidden-file detection on macOS.

Nothing here installs anything: uv, pip and the validators are mocked.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
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
    saved_state,
)
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
        # A real validator report: validate_backend parses its JSON.
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
        spec = Install_Dependencies.backend_specs({"backend": "cpu"})[0]
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
        spec = Install_Dependencies.backend_specs({"backend": "cuda126"})[0]
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
                spec = Install_Dependencies.backend_specs({"backend": backend})[0]
                command = Install_Dependencies.backend_install_commands("uv", python, spec)[0]
                self.assertIn(f"torch=={Install_Dependencies.TORCH_VERSION}", command)
                self.assertTrue(command[-1].endswith(index_suffix))

        mps = Install_Dependencies.backend_specs({"backend": "mps"})[0]
        self.assertNotIn("--index-url", Install_Dependencies.backend_install_commands("uv", python, mps)[0])

    def test_rocm_is_one_architecture_specific_profile(self):
        # The per-ROCm-release profiles are gone: a single `rocm` backend
        # installs from AMD's whl-next channel and selects the device
        # package with the validated GFX target.
        self.assertEqual(
            sorted(
                backend for backend in Install_Dependencies.ACCELERATOR_BACKENDS
                if backend.startswith("rocm")
            ),
            [Install_Dependencies.ROCM_BACKEND],
        )
        spec = Install_Dependencies.backend_specs(
            {"backend": Install_Dependencies.ROCM_BACKEND, "gfx_target": "gfx1100"}
        )[0]
        command = Install_Dependencies.backend_install_commands("uv", Path("python"), spec)[0]
        self.assertIn(
            f"torch[device-gfx1100]=={Install_Dependencies.ROCM_TORCH_VERSION}", command
        )
        self.assertEqual(command[-1], "https://stable.repo.amd.com/rocm/whl-next/")
        # One step on every platform: torch with its device extra is the whole install.
        self.assertEqual(len(spec.install_steps), 1)
        # The recorded version drops the +rocm local segment so it stays
        # comparable with torch.__version__ during runtime validation.
        self.assertEqual(spec.torch_version, Install_Dependencies.TORCH_VERSION)
        with self.assertRaises(ValueError):
            Install_Dependencies.backend_specs(
                {"backend": Install_Dependencies.ROCM_BACKEND}
            )[0]

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
            for line in ("torch==2.14.0", "Transformers >= 5.17.0", "torch"):
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
            return {"validated_devices": [{"spec": "cpu", "success": True}]}

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
                    saved_state(
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
        state = saved_state(specs[0], requirements)
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
        state = saved_state(specs[0], requirements)
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
            state = saved_state(specs[0], requirements)
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
            state = saved_state(specs[0], requirements)
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
            ) as consistency, mock.patch.object(
                Install_Dependencies, "validate_esm_stack", return_value=True
            ), mock.patch.object(Install_Dependencies, "write_state"):
                code = Install_Dependencies.install(
                    project_root=ROOT, venv=venv, uv_executable="uv"
                )

        self.assertEqual(code, 0)
        self.assertEqual(validate.call_args.args[1].backend, "cuda126")
        install_backend.assert_not_called()
        # A setup or repair lists the deviations it accepted; the check before each launch does not.
        consistency.assert_called_once_with("uv", Path(sys.executable), list_accepted=True)

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
                saved_state(
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
                saved_state(
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
                saved_state(
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

    def _install_with(self, report, install_backend):
        success = mock.Mock(returncode=0, stdout="", stderr="")
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(Install_Dependencies, "venv_python", return_value=Path(sys.executable)), \
                mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report), \
                mock.patch.object(Install_Dependencies, "_run", return_value=success), \
                mock.patch.object(Install_Dependencies, "install_backend", side_effect=install_backend), \
                mock.patch.object(Install_Dependencies, "_installed_version", return_value=None), \
                mock.patch.object(Install_Dependencies, "write_state"), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = Install_Dependencies.install(project_root=ROOT, venv=Path(temp_dir), uv_executable="uv")
        return code, stderr.getvalue()

    def test_falling_back_to_cpu_says_why_and_how_to_retry(self):
        def install_backend(_uv, _python, spec):
            if spec.backend == "cpu":
                return {"validated_devices": [{"spec": "cpu", "success": True}]}
            Install_Dependencies._note_backend_failure("torch.cuda.is_available() is false")
            return None

        code, stderr = self._install_with(self._rocm_report(), install_backend)
        self.assertEqual(code, 0)
        self.assertIn("could not be used, so EMAP-SSN will run on the CPU.", stderr)
        self.assertIn("torch.cuda.is_available() is false", stderr)
        self.assertIn("--refresh-backend", stderr)

    def test_a_backend_that_validates_prints_no_fallback_warning(self):
        def install_backend(_uv, _python, _spec):
            return {"validated_devices": [{"spec": "cuda:0", "success": True}]}

        code, stderr = self._install_with(self._rocm_report(), install_backend)
        self.assertEqual(code, 0)
        self.assertNotIn("Warning:", stderr)

    def test_detection_warnings_are_printed(self):
        report = self._cpu_report()
        report["warnings"] = ["NVIDIA GeForce RTX 5090 needs NVIDIA driver 580 or newer (installed: 576.88)."]

        def install_backend(_uv, _python, _spec):
            return {"validated_devices": [{"spec": "cpu", "success": True}]}

        code, stderr = self._install_with(report, install_backend)
        self.assertEqual(code, 0)
        self.assertIn("Warning: NVIDIA GeForce RTX 5090 needs NVIDIA driver 580 or newer", stderr)

    def test_a_failed_tensor_check_reports_each_devices_error(self):
        failed = mock.Mock(
            returncode=0,
            stdout=json.dumps({
                "backend": "cuda132", "profile": "cuda132", "package_error": None,
                "devices": [{"spec": "cuda:0", "name": "NVIDIA GeForce RTX 5090", "success": False,
                             "error": "CUDA error: no kernel image is available for execution on the device"}],
            }),
            stderr="",
        )
        spec = Install_Dependencies.backend_specs({"backend": "cuda132"})[0]
        stderr = io.StringIO()
        with mock.patch.object(Install_Dependencies, "_run", return_value=failed), redirect_stderr(stderr):
            self.assertIsNone(Install_Dependencies.validate_backend(Path("python"), spec))
        self.assertIn("no kernel image is available", stderr.getvalue())
        self.assertIn("no kernel image is available", Install_Dependencies._LAST_BACKEND_FAILURE)

    def test_failed_accelerator_install_falls_back_to_cpu(self):
        report = self._rocm_report()
        success = mock.Mock(returncode=0, stdout="", stderr="")
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(Install_Dependencies, "venv_python", return_value=Path(sys.executable)), \
                mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report), \
                mock.patch.object(Install_Dependencies, "_run", return_value=success), \
                mock.patch.object(
                    Install_Dependencies, "install_backend",
                    side_effect=[None, {"validated_devices": [{"spec": "cpu", "success": True}]}],
                ) as install_backend, \
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
        active = Install_Dependencies.backend_specs({"backend": "cpu"})[0]
        success = mock.Mock(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as temp_dir:
            venv = Path(temp_dir)
            Install_Dependencies.write_state(
                venv / Install_Dependencies.STATE_FILENAME,
                saved_state(active, ROOT / "src" / "requirements.txt"),
                report,
            )
            with mock.patch.object(
                    Install_Dependencies, "venv_python", return_value=Path(sys.executable)
                ), mock.patch.object(
                    Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
                ), mock.patch.object(
                    Install_Dependencies, "_run", return_value=success
                ), mock.patch.object(
                    Install_Dependencies,
                    "validate_backend",
                    return_value={"validated_devices": [{"spec": "cpu", "success": True}]},
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


class DependencyReadinessTests(unittest.TestCase):
    def test_ready_environment_is_checked_without_installing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python = root / "python"
            python.touch()
            state = {"active_backend": {"backend": "cpu"}}
            active = Install_Dependencies.backend_specs({"backend": "cpu"})[0]
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

    def test_without_uv_the_check_skips_only_the_package_consistency_step_and_says_so(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python = root / "python"
            python.touch()
            state = {"active_backend": {"backend": "cpu"}}
            active = Install_Dependencies.backend_specs({"backend": "cpu"})[0]
            stdout = io.StringIO()
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
                    ), redirect_stdout(stdout):
                ready = Install_Dependencies.environment_is_ready(
                    project_root=root, venv=root, uv_executable=None
                )

        self.assertTrue(ready)
        consistency.assert_not_called()
        self.assertEqual(
            stdout.getvalue().count("Installed package consistency: skipped because uv is not installed\n"), 1
        )

    def test_a_gpu_not_visible_now_prints_no_state_mismatch(self):
        # With no accelerator visible, a changed hardware fingerprint keeps the
        # saved build. It used to list "State mismatch" lines on every start
        # anyway; only a mismatch that stops the launch is listed now.
        active = Install_Dependencies.backend_specs({"backend": "cpu"})[0]
        for mismatches, expected_ready in (
            (["hardware_fingerprint", "requested_candidates"], True),
            (["hardware_fingerprint", "requirements"], False),
        ):
            stdout, stderr = io.StringIO(), io.StringIO()
            with self.subTest(mismatches=mismatches), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                python = root / "python"
                python.touch()
                with mock.patch.object(Install_Dependencies, "venv_python", return_value=python), \
                        mock.patch.object(Install_Dependencies, "verify_esm_runtime_requirements"), \
                        mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value={}), \
                        mock.patch.object(Install_Dependencies, "backend_specs", return_value=[]), \
                        mock.patch.object(Install_Dependencies, "hardware_fingerprint", return_value="fp"), \
                        mock.patch.object(Install_Dependencies, "read_state",
                                          return_value={"active_backend": {"backend": "cpu"}}), \
                        mock.patch.object(Install_Dependencies, "_state_mismatches", return_value=list(mismatches)), \
                        mock.patch.object(Install_Dependencies, "_backend_from_state", return_value=active), \
                        mock.patch.object(Install_Dependencies, "_accelerator_runtime_visible", return_value=False), \
                        mock.patch.object(Install_Dependencies, "validate_backend", return_value={"devices": []}), \
                        mock.patch.object(Install_Dependencies, "_installed_version", side_effect=lambda _python, package: (
                            Install_Dependencies.TRANSFORMERS_VERSION if package == "transformers"
                            else Install_Dependencies.ESM_VERSION
                        )), \
                        mock.patch.object(Install_Dependencies, "validate_package_consistency", return_value=True), \
                        mock.patch.object(Install_Dependencies, "validate_esm_stack", return_value=True), \
                        redirect_stdout(stdout), redirect_stderr(stderr):
                    ready = Install_Dependencies.environment_is_ready(
                        project_root=root, venv=root, uv_executable="uv"
                    )
                self.assertEqual(ready, expected_ready)
                if expected_ready:
                    self.assertNotIn("State mismatch", stderr.getvalue())
                    self.assertIn("compatible accelerator state is being preserved", stdout.getvalue())
                else:
                    self.assertIn("State mismatch [requirements]", stderr.getvalue())

    def test_readiness_reports_package_consistency_once(self):
        # A real Linux check printed "Installed package consistency" twice:
        # with the accepted esm deviations, then a bare "passed" that really
        # stood for the ESM import check after it.
        deviations = "\n".join((
            "Found 2 incompatibilities",
            TORCH_DEVIATION_LINE,
            TRANSFORMERS_DEVIATION_LINE,
        ))
        for label, completed in (
            ("esm deviations", mock.Mock(returncode=1, stdout=deviations, stderr="")),
            ("clean", mock.Mock(returncode=0, stdout="", stderr="")),
        ):
            stdout = io.StringIO()
            with self.subTest(report=label), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                python = root / "python"
                python.touch()
                active = Install_Dependencies.backend_specs({"backend": "cpu"})[0]
                with mock.patch.object(Install_Dependencies, "venv_python", return_value=python), \
                        mock.patch.object(Install_Dependencies, "verify_esm_runtime_requirements"), \
                        mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value={}), \
                        mock.patch.object(Install_Dependencies, "backend_specs", return_value=[]), \
                        mock.patch.object(Install_Dependencies, "hardware_fingerprint", return_value="fp"), \
                        mock.patch.object(Install_Dependencies, "read_state",
                                          return_value={"active_backend": {"backend": "cpu"}}), \
                        mock.patch.object(Install_Dependencies, "_state_mismatches", return_value=[]), \
                        mock.patch.object(Install_Dependencies, "_backend_from_state", return_value=active), \
                        mock.patch.object(Install_Dependencies, "validate_backend", return_value={"devices": []}), \
                        mock.patch.object(Install_Dependencies, "_installed_version", side_effect=lambda _python, package: (
                            Install_Dependencies.TRANSFORMERS_VERSION if package == "transformers"
                            else Install_Dependencies.ESM_VERSION
                        )), \
                        mock.patch.object(Install_Dependencies, "_run", return_value=completed), \
                        mock.patch.object(Install_Dependencies, "validate_esm_stack", return_value=True), \
                        redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                    self.assertTrue(Install_Dependencies.environment_is_ready(
                        project_root=root, venv=root, uv_executable="uv"
                    ))
                lines = stdout.getvalue().splitlines()
                self.assertEqual(
                    len([line for line in lines if line.startswith("Installed package consistency")]), 1, lines
                )
                self.assertIn("ESM/Transformers import validation: passed", lines)
                self.assertEqual([line for line in lines if "accepted:" in line], [], lines)  # not on every launch

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


class HiddenEnvironmentTests(unittest.TestCase):
    """macOS marks files of an iCloud-synced folder hidden (UF_HIDDEN); Qt and .pth processing then skip them."""

    HIDDEN = 0x8000

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.venv = Path(temporary.name) / ".venv"
        site_packages = self.venv / "lib" / "python3.13" / "site-packages"
        plugins = site_packages / "PySide6" / "Qt" / "plugins" / "platforms"
        plugins.mkdir(parents=True)
        self.files = {
            "cfg": self.venv / "pyvenv.cfg",
            "pth": site_packages / "_virtualenv.pth",
            "qt": plugins / "libqcocoa.dylib",
        }
        for path in self.files.values():
            path.write_text("x", encoding="utf-8")

    def _check(self, flags_by_name, platform_name="darwin"):
        flags = {path: flags_by_name.get(name, 0) for name, path in self.files.items()}

        def fake_stat(path):
            return mock.Mock(st_flags=flags[Path(path)])

        return Install_Dependencies.hidden_environment_files(self.venv, platform_name, fake_stat)

    def test_unflagged_files_are_not_hidden(self):
        self.assertEqual(self._check({}), [])

    def test_each_sentinel_that_is_flagged_is_reported(self):
        for name in self.files:
            with self.subTest(name=name):
                self.assertEqual(self._check({name: self.HIDDEN | 0x2}), [self.files[name]])

    def test_other_flags_do_not_count(self):
        self.assertEqual(self._check({"cfg": 0x2, "qt": 0x1}), [])

    def test_only_macos_has_the_flag(self):
        self.assertEqual(self._check({"cfg": self.HIDDEN}, "linux"), [])
        self.assertEqual(self._check({"cfg": self.HIDDEN}, "win32"), [])

    def test_a_stat_result_without_flags_or_a_missing_file_is_not_hidden(self):
        class NoFlags:
            pass

        self.assertEqual(
            Install_Dependencies.hidden_environment_files(self.venv, "darwin", lambda path: NoFlags()), [])

        def missing(path):
            raise FileNotFoundError(path)

        self.assertEqual(Install_Dependencies.hidden_environment_files(self.venv, "darwin", missing), [])

    def test_the_readiness_check_stops_with_its_own_code_and_says_what_to_do(self):
        stderr = io.StringIO()
        with mock.patch.object(Install_Dependencies, "hidden_environment_files", return_value=[Path("x")]), \
                mock.patch.object(Install_Dependencies, "_refuse_unsupported_host", return_value=False), \
                mock.patch.object(Install_Dependencies, "environment_is_ready", return_value=True) as ready, \
                mock.patch.object(Install_Dependencies, "install", return_value=0) as install, \
                redirect_stderr(stderr):
            code = Install_Dependencies.main(["--check-only", "--venv", str(self.venv)])
        self.assertEqual(code, Install_Dependencies.HIDDEN_ENVIRONMENT_EXIT)
        self.assertNotIn(code, (0, Install_Dependencies.SETUP_REQUIRED_EXIT, 1))
        self.assertIn("iCloud", stderr.getvalue())
        self.assertIn("run ./install.sh again", stderr.getvalue())
        ready.assert_not_called()
        install.assert_not_called()

    def test_setup_itself_is_not_blocked(self):
        with mock.patch.object(Install_Dependencies, "hidden_environment_files", return_value=[Path("x")]), \
                mock.patch.object(Install_Dependencies, "_refuse_unsupported_host", return_value=False), \
                mock.patch.object(Install_Dependencies.shutil, "which", return_value="/stub/uv"), \
                mock.patch.object(Install_Dependencies, "install", return_value=0) as install:
            code = Install_Dependencies.main(["--venv", str(self.venv)])
        self.assertEqual(code, 0)
        install.assert_called_once()


if __name__ == "__main__":
    unittest.main()
