# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""MCP Compute_Capabilities: the device metadata report (built without running
any computation) and the helper-subprocess wrapper with its tool restrictions."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mcp_server.pipeline import Compute_Capabilities as compute
from utilities import BLAST_Executables as blast
from utilities import Hardware_Acceleration as hardware


def fake_runtime(backend="cuda", count=1):
    def module(available):
        return NS(is_available=lambda: available, device_count=lambda: count,
                  get_device_name=lambda i: "Test GPU", get_device_properties=lambda i: NS(major=8, total_memory=1000),
                  mem_get_info=lambda i: (600, 1000), Stream=lambda: None,
                  Event=lambda: None, stream=lambda: None)
    forbidden = mock.Mock(side_effect=AssertionError("Computation is forbidden"))
    return NS(__version__="test", version=NS(cuda="12" if backend == "cuda" else None,
              hip="6" if backend == "rocm" else None),
              device=lambda spec: spec, cuda=module(backend in ("cuda", "rocm")),
              xpu=module(backend == "xpu"),
              backends=NS(mps=NS(is_available=lambda: backend == "mps", get_name=lambda: "Apple")),
              mps=NS(recommended_max_memory=lambda: 900, driver_allocated_memory=lambda: 100),
              tensor=forbidden, arange=forbidden, mm=forbidden, empty=forbidden,
              synchronize=forbidden, empty_cache=forbidden)


class ComputeCapabilityTests(unittest.TestCase):
    def setUp(self):
        # Where BLAST+ is on this machine is not what these tests are about, and its version query
        # would be one more subprocess.run while they stand in for the helper's.
        self.enterContext(mock.patch.object(compute, "blast_report", return_value={"status": "not_found"}))

    def collect(self, runtime, approved=None):
        with mock.patch.object(hardware, "torch", runtime), \
                mock.patch.object(hardware, "_validated_device_specs", return_value=approved):
            return compute.collect_metadata(runtime, hardware,
                NS(cpu_count=lambda **kw: 4, virtual_memory=lambda: NS(total=2000, available=800)))

    def test_backends_identifiers_memory_and_unverified_capabilities(self):
        for backend in ("cpu", "cuda", "rocm", "xpu", "mps"):
            with self.subTest(backend=backend):
                runtime = fake_runtime(backend)
                result = self.collect(runtime)
                self.assertEqual(result["status"], "ok")
                device = result["devices"][-1]
                self.assertEqual(device["backend"], backend)
                spec = "cuda:0" if backend in ("cuda", "rocm") else "xpu:0" if backend == "xpu" else backend
                self.assertEqual(device["device_selection"], spec)
                self.assertFalse(any(c["computation_verified"] for c in device["capabilities"].values()))
                self.assertEqual(device["capabilities"]["tf32"]["status"], "supported" if backend == "cuda" else "unsupported")
                if backend == "mps":
                    self.assertEqual(device["memory"]["kind"], "unified_working_set")
                    self.assertIsNone(device["memory"]["available_bytes"])
                    self.assertEqual(device["memory"]["recommended_working_set_bytes"], 900)
                elif backend != "cpu":
                    self.assertEqual(device["memory"]["available_bytes"], 600)
                runtime.mm.assert_not_called()
                runtime.arange.assert_not_called()
                runtime.empty.assert_not_called()

    def test_multiple_devices_and_installer_filtering(self):
        result = self.collect(fake_runtime(count=3), approved={"cuda:1", "cuda:2"})
        self.assertEqual([d["device_selection"] for d in result["devices"]], ["cpu", "cuda:1", "cuda:2"])
        self.assertTrue(result["installer_filtering"]["applies"])

    def test_older_cuda_does_not_claim_emulated_bf16_is_unavailable(self):
        runtime = fake_runtime()
        runtime.cuda.get_device_properties = lambda i: NS(major=7, total_memory=1000)
        caps = self.collect(runtime)["devices"][-1]["capabilities"]
        self.assertEqual(caps["tf32"]["status"], "unsupported")
        self.assertEqual(caps["bf16"]["status"], "unknown")

    def test_failures_are_partial_not_false_cpu_only_success(self):
        runtime = fake_runtime()
        runtime.cuda.is_available = mock.Mock(side_effect=RuntimeError("backend failed"))
        result = self.collect(runtime)
        self.assertEqual(result["status"], "partial")
        self.assertIn("backend failed", str(result["errors"]))
        runtime = fake_runtime()
        runtime.cuda.mem_get_info = mock.Mock(side_effect=RuntimeError("memory unavailable"))
        runtime.cuda.get_device_properties = mock.Mock(side_effect=RuntimeError("properties unavailable"))
        result = self.collect(runtime)
        device = result["devices"][-1]
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(device["memory"]["available_bytes"])
        self.assertEqual(device["capabilities"]["bf16"]["status"], "unknown")
        del runtime.cuda.Stream
        self.assertEqual(self.collect(runtime)["devices"][-1]["capabilities"]["tiled"]["status"], "unsupported")

    def test_tool_specific_settings_and_restrictions(self):
        report = self.collect(fake_runtime("mps"))
        completed = NS(returncode=0, stdout=json.dumps(report), stderr="")
        with mock.patch.object(compute.subprocess, "run", return_value=completed) as run:
            result = compute.discover_compute_capabilities(ROOT, "network_injection")
            self.assertEqual(result["devices"][-1]["tool_settings"]["EXECUTION_MODE"]["tiled"]["status"], "unsupported")
            result = compute.discover_compute_capabilities(ROOT, "align_similarity_matrix")
            self.assertEqual(result["devices"][-1]["tool_settings"]["EXECUTION_MODE"]["tiled"]["status"], "supported")
            result = compute.discover_compute_capabilities(ROOT, "sanitize_sequences")
            self.assertEqual(result["tool"]["settings"], {})
            self.assertEqual(result["devices"][-1]["tool_settings"], {})
            self.assertEqual(run.call_args.kwargs["timeout"], 20)
            self.assertEqual(run.call_args.args[0][0], sys.executable)
            run.reset_mock()
            with self.assertRaises(KeyError): compute.discover_compute_capabilities(ROOT, "bogus")
            run.assert_not_called()

    def test_bad_helpers_and_timeout_are_unavailable(self):
        for output in ("not json", "[]", '{"status":"ok","devices":[{}]}'):
            with mock.patch.object(compute.subprocess, "run", return_value=NS(returncode=0, stdout=output, stderr="")):
                self.assertEqual(compute.discover_compute_capabilities(ROOT)["status"], "unavailable")
        with mock.patch.object(compute.subprocess, "run", side_effect=subprocess.TimeoutExpired("helper", 20)):
            report = compute.discover_compute_capabilities(ROOT)
            self.assertEqual(report["status"], "unavailable")
            self.assertEqual(report["devices"], [])

    def test_timeout_reaps_only_the_helper(self):
        real_run, real_popen = subprocess.run, subprocess.Popen
        children = []
        def popen(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child
        def run(*args, **kwargs):
            return real_run([sys.executable, "-c", "import time; time.sleep(5)"],
                            capture_output=True, timeout=0.1)
        with mock.patch.object(subprocess, "Popen", side_effect=popen), \
                mock.patch.object(subprocess, "run", side_effect=run):
            result = compute.discover_compute_capabilities(ROOT)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())


class BlastCapabilityTests(unittest.TestCase):
    """The report's `blast` entry: the BLAST+ a blank BLASTP_DIR finds, and its version."""

    found = blast.BlastExecutables(
        os.path.join("ncbi", "bin", "blastp"), os.path.join("ncbi", "bin", "makeblastdb"), "PATH"
    )

    def test_a_blast_that_is_found_reports_its_version_paths_and_source(self):
        with mock.patch.object(blast, "find_automatic", return_value=self.found), \
                mock.patch.object(blast, "cached_version", return_value="blastp: 2.17.0+") as version:
            report = compute.blast_report()
        self.assertEqual(report["status"], "found")
        self.assertEqual(report["version"], "blastp: 2.17.0+")
        self.assertEqual(report["blastp"], self.found.blastp)
        self.assertEqual(report["makeblastdb"], self.found.makeblastdb)
        self.assertEqual(report["source"], "PATH")
        version.assert_called_once_with(self.found.blastp, compute.VERSION_TIMEOUT)
        self.assertLessEqual(compute.VERSION_TIMEOUT, 10)

    def test_a_blast_in_a_standard_folder_says_so(self):
        standard = blast.BlastExecutables("a", "b", "C:\\NCBI\\blast-2.17.0+\\bin (standard install folder)")
        with mock.patch.object(blast, "find_automatic", return_value=standard), \
                mock.patch.object(blast, "cached_version", return_value="blastp: 2.17.0+"):
            self.assertEqual(compute.blast_report()["source"], standard.source)

    def test_no_blast_says_where_the_user_can_set_or_download_one(self):
        with mock.patch.object(blast, "find_automatic", return_value=None), \
                mock.patch.object(blast, "cached_version", side_effect=AssertionError("ran")):
            report = compute.blast_report()
        self.assertEqual(report["status"], "not_found")
        self.assertIsNone(report["version"])
        self.assertIsNone(report["blastp"])
        for text in ("BLASTP Directory", "Tools window", "Download button", "BLASTP_DIR"):
            self.assertIn(text, report["reason"])

    def test_a_blast_that_cannot_report_its_version_is_found_but_an_error(self):
        with mock.patch.object(blast, "find_automatic", return_value=self.found), \
                mock.patch.object(blast, "cached_version", side_effect=blast.BlastError("it will not start")):
            report = compute.blast_report()
        self.assertEqual(report["status"], "error")
        self.assertIsNone(report["version"])
        self.assertEqual(report["blastp"], self.found.blastp)
        self.assertIn("it will not start", report["reason"])

    def test_blast_runs_once_per_file_however_often_the_report_is_asked_for(self):
        with tempfile.TemporaryDirectory() as folder:
            blastp = os.path.join(folder, "blastp")
            with open(blastp, "wb") as handle:
                handle.write(b"blastp")
            found = blast.BlastExecutables(blastp, os.path.join(folder, "makeblastdb"), "PATH")
            completed = NS(returncode=0, stdout="blastp: 2.17.0+\n", stderr="")
            with mock.patch.object(blast, "find_automatic", return_value=found), \
                    mock.patch.object(subprocess, "run", return_value=completed) as run:
                reports = [compute.blast_report() for _ in range(3)]
        self.assertEqual([report["version"] for report in reports], ["blastp: 2.17.0+"] * 3)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], [blastp, "-version"])
        self.assertEqual(run.call_args.kwargs["timeout"], compute.VERSION_TIMEOUT)
        self.assertEqual(run.call_args.kwargs["env"]["BLAST_USAGE_REPORT"], "false")

    def test_the_capabilities_report_carries_the_blast_entry(self):
        entry = {"status": "found", "version": "blastp: 2.17.0+"}
        report = ComputeCapabilityTests().collect(fake_runtime())
        with mock.patch.object(compute.subprocess, "run", return_value=NS(returncode=0, stdout=json.dumps(report), stderr="")), \
                mock.patch.object(compute, "blast_report", return_value=entry):
            result = compute.discover_compute_capabilities(ROOT)
        self.assertEqual(result["blast"], entry)
        self.assertTrue(any("blast" in note for note in result["notes"]))

    def test_the_blast_entry_is_there_when_the_helper_fails(self):
        entry = {"status": "not_found"}
        with mock.patch.object(compute.subprocess, "run", side_effect=subprocess.TimeoutExpired("helper", 20)), \
                mock.patch.object(compute, "blast_report", return_value=entry):
            result = compute.discover_compute_capabilities(ROOT)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["blast"], entry)


if __name__ == "__main__":
    unittest.main()
