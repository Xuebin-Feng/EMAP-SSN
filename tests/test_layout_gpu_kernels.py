"""utilities.Layout_GPU_Kernels: the runtime-compiled repulsion and spring kernels
of SSNSimulationGPU (against float64 PyTorch and an independent NumPy reference,
in 2D and 3D, across both force caps, the taper, inactive nodes, components,
split partner ranges and tile skipping), their determinism, the PyTorch fallback,
NVIDIA/ROCm library and compiler-option selection, the self-check and the
SSN_LAYOUT_GPU_KERNELS switch."""
import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Layout_Engine_SSN as ssn_engine

HAS_TORCH = ssn_engine.HAS_TORCH
if HAS_TORCH:
    import torch
    from utilities import Layout_GPU_Kernels as kernels_module

GPU_AVAILABLE = HAS_TORCH and torch.cuda.is_available()
FORCE_PARAMS = {
    "COULOMB_K": 10.0,
    "MAX_FORCE_LIMIT": 20.0,
    "COULOMB_CUTOFF": 15.0,
    "SPRING_K": 5.0,
    "DAMPING": 0.9,
    "DT": 0.005,
}
# Correct float32 kernels stay near 4e-6 of the largest force (RTX 4070).
TOLERANCE = 5e-5


# (nodes, spread, blob centres on the first axis) per component: tight (capped
# pairs), taper-sized, two blobs 240 apart and small. The blobs are big enough to
# fill whole 128-node tiles, so some tiles of one component lie beyond the
# cutoff of each other (a tile shared with a neighbour spans both).
DEFAULT_COMPONENTS = (
    (900, 0.4, (0.0,)),
    (600, 7.0, (0.0,)),
    (600, 10.0, (-120.0, 120.0)),
    (40, 2.0, (0.0,)),
)


def make_problem(seed, dims, components=DEFAULT_COMPONENTS):
    """Components interleaved in node order, with a tenth of the nodes inactive."""
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for label, (size, spread, centres) in enumerate(components):
        for centre, part in zip(centres, np.array_split(np.arange(size), len(centres))):
            offset = np.zeros(dims)
            offset[0] = centre
            blocks.append(rng.normal(0.0, spread, (len(part), dims)) + offset)
            labels.append(np.full(len(part), label))
    pos = np.concatenate(blocks).astype(np.float32)
    comp = np.concatenate(labels).astype(np.int64)
    sizes = [size for size, _, _ in components]
    order = rng.permutation(len(pos))
    pos, comp = pos[order], comp[order]
    active = rng.random(len(pos)) > 0.1
    edges = []
    for label in range(len(sizes)):
        members = np.flatnonzero(comp == label)
        edges.append(np.column_stack((rng.choice(members, 4 * len(members)),
                                      rng.choice(members, 4 * len(members)))))
    return pos, comp, active, np.concatenate(edges).astype(np.int64)


def numpy_repulsion(pos, comp, active, params):
    """Independent float64 all-pairs reference with the engine's physics."""
    pos = pos.astype(np.float64)
    out = np.zeros_like(pos)
    cutoff = params["COULOMB_CUTOFF"]
    nodes = np.flatnonzero(active)
    for a in nodes:
        for b in nodes:
            if a == b or comp[a] != comp[b]:
                continue
            delta = pos[a] - pos[b]
            distance = float(np.sqrt(delta @ delta))
            if distance == 0.0 or distance > cutoff:
                continue
            force = params["COULOMB_K"] / max(distance, 0.5) ** 2
            if params["MAX_FORCE_LIMIT"] > 0.0:
                force = min(force, params["MAX_FORCE_LIMIT"])
            if distance > cutoff * 0.8:
                force *= min(max((cutoff - distance) / (cutoff * 0.2), 0.0), 1.0)
            out[a] += force * delta / distance
    cap = params.get("MAX_TOTAL_REPULSION_FORCE", 0.0)
    if cap > 0.0:
        norms = np.linalg.norm(out, axis=1)
        scale = np.minimum(cap / np.maximum(norms, 1e-12), 1.0)
        out *= scale[:, None]
    return out


def numpy_springs(pos, active, edges, spring_k):
    pos = pos.astype(np.float64)
    out = np.zeros_like(pos)
    for a, b in edges:
        if not (active[a] and active[b]):
            continue
        delta = pos[a] - pos[b]
        length = np.sqrt(delta @ delta) + 1e-9
        force = -spring_k * length * delta / length
        out[a] += force
        out[b] -= force
    return out


def assert_close(test, got, expected, what):
    got = np.asarray(got, dtype=np.float64)
    scale = max(float(np.abs(expected).max()), 1.0)
    error = float(np.abs(got - expected).max())
    test.assertLessEqual(error, TOLERANCE * scale,
                         f"{what}: max error {error:.3g} against max force {scale:.3g}")


@unittest.skipUnless(HAS_TORCH, "PyTorch is unavailable")
class TorchFallbackTests(unittest.TestCase):
    """The PyTorch formulas used where no kernels run (and as the reference)."""

    def test_repulsion_matches_independent_reference_with_both_caps(self):
        pos, comp, active, _ = make_problem(
            1, 2, components=((12, 0.3, (0.0,)), (10, 6.0, (0.0,)), (8, 4.0, (0.0,))))
        for cap in (0.0, 4.0):
            params = dict(FORCE_PARAMS, MAX_TOTAL_REPULSION_FORCE=cap)
            got = kernels_module.torch_repulsion(
                torch.tensor(pos, dtype=torch.float64), torch.tensor(comp),
                torch.tensor(active), params).numpy()
            np.testing.assert_allclose(got, numpy_repulsion(pos, comp, active, params),
                                       rtol=1e-9, atol=1e-9)
            self.assertTrue(np.all(got[~active] == 0.0))

    def test_row_blocks_do_not_change_the_result(self):
        pos, comp, active, _ = make_problem(
            2, 3, components=((40, 1.0, (0.0,)), (30, 8.0, (0.0,))))
        tensors = (torch.tensor(pos, dtype=torch.float64), torch.tensor(comp), torch.tensor(active))
        params = dict(FORCE_PARAMS, MAX_TOTAL_REPULSION_FORCE=2.0)
        whole = kernels_module.torch_repulsion(*tensors, params, rows=len(pos))
        for rows in (1, 7, None):
            np.testing.assert_allclose(
                kernels_module.torch_repulsion(*tensors, params, rows=rows), whole,
                rtol=1e-12, atol=1e-12)

    def test_default_row_blocks_bound_the_pairs_evaluated_at_once(self):
        self.assertEqual(kernels_module.chunk_rows(100), 100)
        self.assertEqual(kernels_module.chunk_rows(44127), kernels_module.CHUNK_PAIRS // 44127)
        self.assertLessEqual(kernels_module.chunk_rows(44127) * 44127, kernels_module.CHUNK_PAIRS)

    def test_springs_skip_inactive_ends_and_match_reference(self):
        pos, comp, active, edges = make_problem(
            3, 2, components=((20, 3.0, (0.0,)), (15, 3.0, (0.0,))))
        springs = kernels_module.TorchSprings(edges, active, torch.device("cpu"))
        got = springs.add(torch.tensor(pos, dtype=torch.float64),
                          torch.zeros(pos.shape, dtype=torch.float64), 5.0).numpy()
        np.testing.assert_allclose(got, numpy_springs(pos, active, edges, 5.0),
                                   rtol=1e-9, atol=1e-9)

    def test_cpu_simulation_uses_the_fallback(self):
        pos, comp, active, edges = make_problem(4, 2, components=((10, 2.0, (0.0,)),))
        simulation = ssn_engine.SSNSimulationGPU(pos, edges, comp, 100.0, FORCE_PARAMS, active)
        self.assertFalse(simulation.uses_kernels)
        self.assertIsNone(kernels_module.layout_kernels(torch.device("cpu"), 2))
        self.assertIn("cpu devices", kernels_module.fallback_reason(torch.device("cpu"), 2))


@unittest.skipUnless(HAS_TORCH, "PyTorch is unavailable")
class RuntimeSelectionTests(unittest.TestCase):
    """Library names and compiler options for NVIDIA (NVRTC) and ROCm (hiprtc)."""

    def _candidates(self, function, *, hip, cuda, platform, sdk_path=None):
        fake_sdk = types.SimpleNamespace(find_libraries=lambda name: [f"{sdk_path}/{name}"])
        modules = {"rocm_sdk": fake_sdk} if sdk_path else {"rocm_sdk": None}
        with mock.patch.object(torch.version, "hip", hip), \
                mock.patch.object(torch.version, "cuda", cuda), \
                mock.patch.object(kernels_module, "_package_library_globs", return_value=[]), \
                mock.patch.dict(sys.modules, modules):
            return function(platform=platform)

    def test_nvidia_names(self):
        windows = self._candidates(kernels_module.rtc_library_candidates,
                                   hip=None, cuda="13.2", platform="win32")
        linux = self._candidates(kernels_module.rtc_library_candidates,
                                 hip=None, cuda="12.8", platform="linux")
        self.assertEqual(windows[0], "nvrtc64_130_0.dll")
        self.assertEqual(linux[:2], ["libnvrtc.so.12", "libnvrtc.so"])
        self.assertEqual(self._candidates(kernels_module.driver_library_candidates,
                                          hip=None, cuda="13.2", platform="win32"), ["nvcuda.dll"])
        self.assertEqual(self._candidates(kernels_module.driver_library_candidates,
                                          hip=None, cuda="13.2", platform="linux")[0], "libcuda.so.1")

    def test_rocm_prefers_the_rocm_sdk_wheels_then_standard_names(self):
        rtc = self._candidates(kernels_module.rtc_library_candidates, hip="7.14.1", cuda=None,
                               platform="win32", sdk_path="C:/sdk")
        driver = self._candidates(kernels_module.driver_library_candidates, hip="7.14.1",
                                  cuda=None, platform="win32", sdk_path="C:/sdk")
        self.assertEqual(rtc[:2], ["C:/sdk/hiprtc", "hiprtc0714.dll"])
        self.assertEqual(driver[:3], ["C:/sdk/amdhip64", "amdhip64_7.dll", "amdhip64.dll"])
        linux_rtc = self._candidates(kernels_module.rtc_library_candidates, hip="7.14.1",
                                     cuda=None, platform="linux")
        linux_driver = self._candidates(kernels_module.driver_library_candidates, hip="7.14.1",
                                        cuda=None, platform="linux")
        self.assertEqual(linux_rtc[0], "libhiprtc.so")
        self.assertEqual(linux_driver[0], "libamdhip64.so")

    def test_compile_options_keep_products_unfused_on_both_vendors(self):
        nvidia = kernels_module.compile_options(3, hip=False, arch="sm_89")
        amd = kernels_module.compile_options(2, hip=True, arch="gfx1100")
        self.assertEqual(nvidia, ["-DDIM=3", "--gpu-architecture=sm_89", "--fmad=false"])
        self.assertEqual(amd, ["-DDIM=2", "--offload-arch=gfx1100", "-ffp-contract=off"])

    def test_kernel_source_avoids_vendor_specific_headers(self):
        source = kernels_module.KERNEL_SOURCE
        self.assertNotIn("#include", source)
        for name in kernels_module.KERNEL_NAMES:
            self.assertIn(f'extern "C" __global__ void {name}(', source)


class _TorchPlan:
    """A repulsion "kernel" made of the float32 PyTorch formulas, optionally wrong."""

    def __init__(self, comp, active, perturbation):
        self.comp = torch.tensor(comp)
        self.active = torch.tensor(active)
        self.perturbation = perturbation

    def skipped_tiles(self, pos, params):
        return 1

    def compute(self, pos, params):
        out = kernels_module.torch_repulsion(pos, self.comp, self.active, params)
        if self.perturbation:
            node = int(torch.nonzero(self.active)[0, 0])
            out[node] += self.perturbation * float(out.abs().max())
        return out


def fake_kernels(perturbation=0.0, *, from_cache=False):
    """A CPU stand-in for LayoutKernels whose self_check is the real one."""
    fake = types.SimpleNamespace(
        device=torch.device("cpu"), dims=2, from_cache=from_cache,
        repulsion_plan=lambda comp, active, count, **options: _TorchPlan(comp, active, perturbation),
        spring_plan=lambda edges, active, count: kernels_module.TorchSprings(
            edges, active, torch.device("cpu")),
        store_in_cache=mock.Mock(), discard_cache=mock.Mock())
    fake.self_check = types.MethodType(kernels_module.LayoutKernels.self_check, fake)
    return fake


@unittest.skipUnless(HAS_TORCH, "PyTorch is unavailable")
class SelfCheckTests(unittest.TestCase):
    """The self-check accepts a correct float32 implementation and rejects errors."""

    def test_correct_float32_forces_pass(self):
        fake_kernels().self_check()

    def test_a_wrong_force_fails_the_check(self):
        with self.assertRaisesRegex(RuntimeError, "self-check failed for mixed repulsion"):
            fake_kernels(1e-3).self_check()

    def test_check_problems_cover_every_case_the_kernels_branch_on(self):
        mixed = kernels_module._self_check_problem(2)
        gap = kernels_module._tile_gap_problem(3)
        active_count = int(mixed["active"].sum())
        self.assertGreater(active_count // kernels_module.TILE, 3)          # several tiles,
        self.assertNotEqual(active_count % kernels_module.TILE, 0)          # a partial last one,
        self.assertEqual(len(np.unique(mixed["comp"])), 3)                  # components,
        self.assertFalse(mixed["active"].all())                             # inactive nodes,
        distances = np.linalg.norm(mixed["pos"][:, None] - mixed["pos"][None], axis=2)
        same = mixed["comp"][:, None] == mixed["comp"][None]
        self.assertTrue((same & (distances > 0) & (distances < 0.7)).any())  # capped pairs,
        self.assertTrue((same & (distances > 12) & (distances < 15)).any())  # the taper zone
        self.assertEqual(len(gap["pos"]), 4 * kernels_module.TILE)


@unittest.skipUnless(HAS_TORCH, "PyTorch is unavailable")
class KernelAvailabilityTests(unittest.TestCase):
    """layout_kernels(): the switch, failures and the self-check gate."""

    def setUp(self):
        patches = [
            mock.patch.dict(kernels_module._KERNELS, clear=True),
            mock.patch.dict(kernels_module._REASONS, clear=True),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(kernels_module, "_device_index", return_value=0),
            mock.patch.object(kernels_module, "_safe_device_name", return_value="Test GPU"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_environment_switch_disables_kernels(self):
        with mock.patch.dict(os.environ, {kernels_module.DISABLE_ENVIRONMENT_VARIABLE: "0"}), \
                mock.patch.object(kernels_module, "LayoutKernels") as constructor:
            self.assertIsNone(kernels_module.layout_kernels(torch.device("cuda", 0), 2))
            reason = kernels_module.fallback_reason(torch.device("cuda", 0), 2)
        constructor.assert_not_called()
        self.assertIn(kernels_module.DISABLE_ENVIRONMENT_VARIABLE, reason)

    def test_compile_failure_falls_back_once_with_the_reason(self):
        output = io.StringIO()
        with mock.patch.object(kernels_module, "LayoutKernels",
                               side_effect=RuntimeError("NVRTC compilation failed")) as constructor, \
                redirect_stdout(output):
            self.assertIsNone(kernels_module.layout_kernels(torch.device("cuda", 0), 2))
            self.assertIsNone(kernels_module.layout_kernels(torch.device("cuda", 0), 2))
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual(output.getvalue().count("GPU layout kernels are unavailable"), 1)
        self.assertIn("NVRTC compilation failed",
                      kernels_module.fallback_reason(torch.device("cuda", 0), 2))
        self.assertIn("PyTorch fallback", kernels_module.describe(torch.device("cuda", 0), 2))

    def test_failed_self_check_keeps_kernels_unused(self):
        fake = mock.Mock(from_cache=False)
        fake.self_check.side_effect = RuntimeError("self-check failed for springs")
        with mock.patch.object(kernels_module, "LayoutKernels", return_value=fake), \
                redirect_stdout(io.StringIO()):
            self.assertIsNone(kernels_module.layout_kernels(torch.device("cuda", 0), 3))
        self.assertIn("self-check failed", kernels_module.fallback_reason(torch.device("cuda", 0), 3))
        fake.store_in_cache.assert_not_called()

    def test_numeric_mismatch_falls_back(self):
        wrong = fake_kernels(1e-3)
        with mock.patch.object(kernels_module, "LayoutKernels", return_value=wrong), \
                redirect_stdout(io.StringIO()):
            self.assertIsNone(kernels_module.layout_kernels(torch.device("cuda", 0), 2))
        self.assertIn("self-check failed for mixed repulsion",
                      kernels_module.fallback_reason(torch.device("cuda", 0), 2))
        wrong.store_in_cache.assert_not_called()

    def test_cached_binary_that_disagrees_is_rebuilt_once(self):
        cached, fresh = fake_kernels(1e-3, from_cache=True), fake_kernels()
        with mock.patch.object(kernels_module, "LayoutKernels",
                               side_effect=[cached, fresh]) as constructor:
            found = kernels_module.layout_kernels(torch.device("cuda", 0), 2)
        self.assertIs(found, fresh)
        cached.discard_cache.assert_called_once()
        self.assertEqual(constructor.call_args_list[1].kwargs, {"read_cache": False})
        fresh.store_in_cache.assert_called_once()

    def test_cache_folder_follows_the_platform_and_the_override(self):
        variable = kernels_module.CACHE_ENVIRONMENT_VARIABLE
        with mock.patch.dict(os.environ, {variable: "", "LOCALAPPDATA": r"C:\Users\u\AppData\Local",
                                          "XDG_CACHE_HOME": "/home/u/.cache"}):
            self.assertEqual(kernels_module.kernel_cache_folder("win32"),
                             os.path.join(r"C:\Users\u\AppData\Local", "EMAP-SSN", "gpu_kernels"))
            self.assertEqual(kernels_module.kernel_cache_folder("linux"),
                             os.path.join("/home/u/.cache", "emap-ssn", "gpu_kernels"))
        with mock.patch.dict(os.environ, {variable: "off"}):
            self.assertIsNone(kernels_module.kernel_cache_folder("linux"))
        with mock.patch.dict(os.environ, {variable: "/scratch/kernels"}):
            self.assertEqual(kernels_module.kernel_cache_folder("win32"), "/scratch/kernels")

    def test_self_check_comparison_rejects_errors_and_nan(self):
        expected = torch.tensor([[100.0, 0.0]], dtype=torch.float64)
        kernels_module._assert_close(expected + 1e-4, expected, "ok")
        with self.assertRaises(RuntimeError):
            kernels_module._assert_close(expected + 0.01, expected, "bad")
        with self.assertRaises(RuntimeError):
            kernels_module._assert_close(expected * float("nan"), expected, "nan")


@unittest.skipUnless(GPU_AVAILABLE, "no CUDA or ROCm GPU")
class KernelForceTests(unittest.TestCase):
    """The compiled kernels against float64 PyTorch on the GPU in this machine."""

    device = torch.device("cuda", 0) if GPU_AVAILABLE else None

    def kernels(self, dims):
        kernels = kernels_module.layout_kernels(self.device, dims)
        self.assertIsNotNone(kernels, kernels_module.fallback_reason(self.device, dims))
        return kernels

    def reference(self, pos, comp, active, params):
        return kernels_module.torch_repulsion(
            torch.tensor(pos, dtype=torch.float64), torch.tensor(comp), torch.tensor(active),
            params).numpy()

    def test_repulsion_matches_float64_in_2d_and_3d(self):
        for dims in (2, 3):
            pos, comp, active, _ = make_problem(10 + dims, dims)
            pos_gpu = torch.tensor(pos, device=self.device)
            for cap in (0.0, 3.0):
                params = dict(FORCE_PARAMS, MAX_TOTAL_REPULSION_FORCE=cap)
                expected = self.reference(pos, comp, active, params)
                for use_skip in (False, True):
                    for j_split in (1, 4, None):
                        plan = self.kernels(dims).repulsion_plan(
                            comp, active, len(pos), use_skip=use_skip, j_split=j_split)
                        if use_skip:
                            self.assertGreater(plan.skipped_tiles(pos_gpu, params), 0)
                        got = plan.compute(pos_gpu, params).cpu().numpy()
                        what = f"dims={dims} cap={cap} skip={use_skip} split={plan.j_split}"
                        assert_close(self, got, expected, what)
                        self.assertTrue(np.all(got[~active] == 0.0), what)

    def test_automatic_split_fills_the_device(self):
        pos, comp, active, _ = make_problem(20, 2)
        plan = self.kernels(2).repulsion_plan(comp, active, len(pos))
        self.assertGreater(plan.j_split, 1)
        self.assertLessEqual(plan.j_split, plan.tile_count)
        self.assertFalse(plan.use_skip)   # below SKIP_MIN_NODES

    def test_tile_skipping_stays_exact_between_resorts(self):
        pos, comp, active, _ = make_problem(21, 2)
        rng = np.random.default_rng(5)
        plan = self.kernels(2).repulsion_plan(comp, active, len(pos), use_skip=True)
        for step in range(2 * kernels_module.RESORT_INTERVAL + 3):
            pos = pos + rng.normal(0.0, 0.8, pos.shape).astype(np.float32)
            got = plan.compute(torch.tensor(pos, device=self.device), FORCE_PARAMS)
            if step % 7 == 0 or step == 2 * kernels_module.RESORT_INTERVAL + 2:
                assert_close(self, got.cpu().numpy(), self.reference(pos, comp, active, FORCE_PARAMS),
                             f"step {step}")

    def test_compiled_binaries_are_cached_and_a_corrupt_file_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(os.environ, {kernels_module.CACHE_ENVIRONMENT_VARIABLE: folder}):
            fresh = kernels_module.LayoutKernels(self.device, 2)
            self.assertFalse(fresh.from_cache)
            fresh.self_check()
            fresh.store_in_cache()
            self.assertTrue(os.path.isfile(fresh.cache_path))
            self.assertTrue(fresh.cache_path.startswith(folder))
            cached = kernels_module.LayoutKernels(self.device, 2)
            self.assertTrue(cached.from_cache)
            self.assertIn("cached", cached.describe())
            cached.self_check()
            with open(fresh.cache_path, "wb") as handle:
                handle.write(b"sm_89\nnot a kernel binary")
            rebuilt = kernels_module.LayoutKernels(self.device, 2)
            self.assertFalse(rebuilt.from_cache)
            self.assertFalse(os.path.exists(fresh.cache_path))
            rebuilt.self_check()

    def test_springs_match_float64(self):
        for dims in (2, 3):
            pos, comp, active, edges = make_problem(30 + dims, dims)
            plan = self.kernels(dims).spring_plan(edges, active, len(pos))
            pos_gpu = torch.tensor(pos, device=self.device)
            got = plan.add(pos_gpu, torch.zeros_like(pos_gpu), 5.0).cpu().numpy()
            expected = kernels_module.TorchSprings(edges, active, torch.device("cpu")).add(
                torch.tensor(pos, dtype=torch.float64),
                torch.zeros(pos.shape, dtype=torch.float64), 5.0).numpy()
            assert_close(self, got, expected, f"springs dims={dims}")


@unittest.skipUnless(GPU_AVAILABLE, "no CUDA or ROCm GPU")
class KernelSimulationTests(unittest.TestCase):
    """SSNSimulationGPU on the GPU: kernels in use, deterministic, same physics."""

    device = torch.device("cuda", 0) if GPU_AVAILABLE else None

    def simulate(self, pos, comp, active, edges, steps, *, box=60.0):
        simulation = ssn_engine.SSNSimulationGPU(pos, edges, comp, box, FORCE_PARAMS, active,
                                                 device=self.device)
        rmsd = [simulation.step(step) for step in range(steps)]
        return simulation, simulation.get_pos(), rmsd

    def test_steps_are_bitwise_reproducible(self):
        pos, comp, active, edges = make_problem(40, 2)
        first, first_pos, first_rmsd = self.simulate(pos, comp, active, edges, 12)
        _, second_pos, second_rmsd = self.simulate(pos, comp, active, edges, 12)
        self.assertTrue(first.uses_kernels)
        np.testing.assert_array_equal(first_pos, second_pos)
        self.assertEqual(first_rmsd, second_rmsd)

    def test_kernels_and_fallback_take_the_same_steps(self):
        for dims in (2, 3):
            pos, comp, active, edges = make_problem(50 + dims, dims)
            fast, fast_pos, fast_rmsd = self.simulate(pos, comp, active, edges, 5)
            with mock.patch.dict(os.environ, {kernels_module.DISABLE_ENVIRONMENT_VARIABLE: "0"}):
                slow, slow_pos, slow_rmsd = self.simulate(pos, comp, active, edges, 5)
            self.assertTrue(fast.uses_kernels)
            self.assertFalse(slow.uses_kernels)
            np.testing.assert_allclose(fast_pos, slow_pos, rtol=0.0, atol=1e-3)
            np.testing.assert_allclose(fast_rmsd, slow_rmsd, rtol=1e-4)
            np.testing.assert_array_equal(fast_pos[~active], pos[~active])

    def test_layout_reports_the_kernels_it_uses(self):
        edges = []
        for base, size in ((0, 12), (12, 12)):
            edges += [(base + i, base + j, 1.0) for i in range(size) for j in range(i + 1, size)]
        params = {"MAX_STEPS": 5, "BOX_SCALE": 1.0, "SIMILARITY_THRESHOLD": 0.0,
                  "PACKING_GRID_SIZE": 200.0, "PACKING_PADDING": 50.0,
                  "LAYOUT_DEVICE_SELECTION": "cuda:0", "LAYOUT_DIMENSIONS": 3}
        output = io.StringIO()
        with redirect_stdout(output):
            positions, _ = ssn_engine.calculate_layout(np.array(edges), 24, params)
        self.assertIn("GPU layout physics on", output.getvalue())
        self.assertIn("runtime-compiled kernels", output.getvalue())
        self.assertTrue(np.isfinite(positions).all())
        self.assertEqual(positions.shape, (24, 3))


if __name__ == "__main__":
    unittest.main()
