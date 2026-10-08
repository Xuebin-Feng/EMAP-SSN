# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Runtime-compiled GPU kernels for the SSN layout physics step.

SSNSimulationGPU's repulsion and spring forces run as two CUDA C kernels that
are compiled the first time a layout uses a GPU: by NVRTC on NVIDIA builds of
PyTorch and by hiprtc on ROCm (AMD) builds. Both compilers ship with those
PyTorch wheels, so no toolkit has to be installed. The repulsion kernel visits
each same-component pair inside registers, so memory grows with nodes plus
springs instead of with nodes squared, and every sum runs in a fixed order, so
a step gives the same result on every run.

Before first use on a device in each process, the kernels are checked against
a float64 evaluation of the same formulas (`torch_repulsion`) on a small fixed
problem that covers both force caps, the cutoff taper, inactive nodes, several
components, split partner ranges, a partial last tile and tile skipping.
Kernels that fail to compile, load or agree are never used: the simulation
falls back to `torch_repulsion` in row blocks and says why once. The AMD path
cannot be tested without AMD hardware, so this check is what stands between an
untested ROCm build and a layout. Setting SSN_LAYOUT_GPU_KERNELS=0 forces the
fallback.

Compiled binaries that passed the check are kept in a per-user cache folder
(%LOCALAPPDATA%\\EMAP-SSN\\gpu_kernels on Windows, ~/.cache/emap-ssn/gpu_kernels
on Linux), keyed by the source, the compiler version and the GPU architecture.
SSN_LAYOUT_GPU_KERNEL_CACHE names another folder, or turns the cache off with 0.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import math
import os
import re
import sys
import tempfile
import threading

import numpy as np
import torch


DISABLE_ENVIRONMENT_VARIABLE = "SSN_LAYOUT_GPU_KERNELS"
CACHE_ENVIRONMENT_VARIABLE = "SSN_LAYOUT_GPU_KERNEL_CACHE"
_OFF_VALUES = {"0", "off", "false", "no"}
TILE = 128
# Pairs evaluated at once by the PyTorch fallback (about 0.75 GiB of temporaries).
CHUNK_PAIRS = 1 << 24
# Below this many active nodes, sorting nodes into tiles costs more than the
# tiles it lets the repulsion kernel skip.
SKIP_MIN_NODES = 16384
RESORT_INTERVAL = 20
# Enough blocks for about this many resident 128-thread blocks per multiprocessor.
BLOCKS_PER_MULTIPROCESSOR = 8
SPRING_BLOCK = 256
# Largest self-check error allowed, as a fraction of the largest force. Correct
# kernels reach 4e-6 on an RTX 4070 (2026-10-08); a dropped taper or cap does not.
_SELF_CHECK_TOLERANCE = 2e-5


KERNEL_SOURCE = r"""
#ifndef DIM
#define DIM 2
#endif
#define TILE 128

static __device__ inline int imin_(int a, int b) { return a < b ? a : b; }
static __device__ inline int imax_(int a, int b) { return a > b ? a : b; }

/* All-pairs repulsion of SSNSimulationGPU (COULOMB_K / max(d, 0.5)^2, the
   MAX_FORCE_LIMIT pair cap, the taper over the outer fifth of the cutoff and the
   MAX_TOTAL_REPULSION_FORCE node cap) for active nodes grouped by component.
   One thread per node; blockIdx.y takes one share of the partner tiles. With
   j_split > 1 every share is written to partial[y][node] and summed afterwards
   in a fixed order. Arithmetic follows PyTorch's per-operation rounding:
   scalar / tensor is a reciprocal times the scalar, tensor / scalar multiplies
   by the reciprocal, and the program is compiled without contraction. */
extern "C" __global__ void ssn_repulsion(
    const float* __restrict__ run_pos, const int* __restrict__ run_node,
    const int* __restrict__ run_start, const int* __restrict__ run_end,
    const float* __restrict__ tile_box, float* __restrict__ out,
    float* __restrict__ partial, int n_nodes, int n_run, int j_split, int use_skip,
    double k_coul_d, double max_f_d, double max_total_d, double cutoff_d)
{
    __shared__ float tile[TILE * DIM];
    __shared__ int block_range[2];
    const float k_coul = (float)k_coul_d;
    const float max_f = (float)max_f_d;
    const float max_total = (float)max_total_d;
    const float cutoff = (float)cutoff_d;
    const float cutoff_sq = (float)(cutoff_d * cutoff_d);
    const float taper_start = (float)(cutoff_d * 0.8);
    const float inv_taper_width = 1.0f / (float)fmax(cutoff_d * 0.2, 1e-9);
    /* A tile is skipped only when its box is clearly beyond the cutoff. */
    const float skip_sq = cutoff_sq * 1.0001f + 1e-6f;

    const int p = blockIdx.x * TILE + threadIdx.x;
    const bool valid = p < n_run;
    float mine[DIM];
    int my_start = 0, my_end = 0;
    for (int d = 0; d < DIM; ++d) mine[d] = valid ? run_pos[p * DIM + d] : 0.0f;
    if (valid) {
        my_start = run_start[p];
        my_end = run_end[p];
    }
    if (threadIdx.x == 0) {
        const int first = blockIdx.x * TILE;
        block_range[0] = run_start[first];
        block_range[1] = run_end[imin_(first + TILE, n_run) - 1];
    }
    __syncthreads();
    const int t0 = block_range[0] / TILE;
    const int t1 = (block_range[1] + TILE - 1) / TILE;
    const int share = (t1 - t0 + j_split - 1) / j_split;
    const int ta = t0 + (int)blockIdx.y * share;
    const int tb = imin_(t1, ta + share);

    float force[DIM];
    for (int d = 0; d < DIM; ++d) force[d] = 0.0f;
    for (int t = ta; t < tb; ++t) {
        if (use_skip) {
            /* Both boxes are the same for every thread, so the block agrees. */
            float gap_sq = 0.0f;
            for (int d = 0; d < DIM; ++d) {
                const float lo_a = tile_box[blockIdx.x * 2 * DIM + d];
                const float hi_a = tile_box[blockIdx.x * 2 * DIM + DIM + d];
                const float lo_b = tile_box[t * 2 * DIM + d];
                const float hi_b = tile_box[t * 2 * DIM + DIM + d];
                const float gap = fmaxf(0.0f, fmaxf(lo_b - hi_a, lo_a - hi_b));
                gap_sq += gap * gap;
            }
            if (gap_sq > skip_sq) continue;
        }
        const int q0 = t * TILE;
        __syncthreads();
        if (q0 + (int)threadIdx.x < n_run) {
            for (int d = 0; d < DIM; ++d)
                tile[threadIdx.x * DIM + d] = run_pos[(q0 + threadIdx.x) * DIM + d];
        }
        __syncthreads();
        if (valid) {
            const int k_lo = imax_(0, my_start - q0);
            const int k_hi = imin_(imin_(TILE, n_run - q0), my_end - q0);
            /* Two-level sum: a tile's pairs first, then the tile into the total. */
            float part[DIM];
            for (int d = 0; d < DIM; ++d) part[d] = 0.0f;
            for (int k = k_lo; k < k_hi; ++k) {
                float delta[DIM];
                for (int d = 0; d < DIM; ++d) delta[d] = mine[d] - tile[k * DIM + d];
                float dist_sq = delta[0] * delta[0];
                for (int d = 1; d < DIM; ++d) dist_sq += delta[d] * delta[d];
                if (dist_sq > 0.0f && dist_sq <= cutoff_sq) {
                    const float dist = sqrtf(dist_sq);
                    const float safe = fmaxf(dist, 0.5f);
                    float f = (1.0f / (safe * safe)) * k_coul;
                    if (max_f > 0.0f) f = fminf(f, max_f);
                    if (dist > taper_start) {
                        f = f * fminf(fmaxf((cutoff - dist) * inv_taper_width, 0.0f), 1.0f);
                    }
                    for (int d = 0; d < DIM; ++d) part[d] += f * (delta[d] / dist);
                }
            }
            for (int d = 0; d < DIM; ++d) force[d] += part[d];
        }
    }
    if (!valid) return;
    const int node = run_node[p];
    if (j_split > 1) {
        for (int d = 0; d < DIM; ++d)
            partial[((long long)blockIdx.y * n_nodes + node) * DIM + d] = force[d];
        return;
    }
    if (max_total > 0.0f) {
        float norm_sq = force[0] * force[0];
        for (int d = 1; d < DIM; ++d) norm_sq += force[d] * force[d];
        const float scale = fminf((1.0f / fmaxf(sqrtf(norm_sq), 1e-12f)) * max_total, 1.0f);
        for (int d = 0; d < DIM; ++d) force[d] = force[d] * scale;
    }
    for (int d = 0; d < DIM; ++d) out[node * DIM + d] = force[d];
}

/* Zero-rest-length springs of SSNSimulationGPU, summed per node over a
   symmetric CSR list of its active springs (no atomics) and added to acc. */
extern "C" __global__ void ssn_springs(
    const float* __restrict__ pos, const long long* __restrict__ row_ptr,
    const int* __restrict__ col, float* __restrict__ acc, int n_nodes, double k_spr_d)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    const float neg_k = (float)(-k_spr_d);
    float mine[DIM], total[DIM];
    for (int d = 0; d < DIM; ++d) {
        mine[d] = pos[i * DIM + d];
        total[d] = 0.0f;
    }
    for (long long e = row_ptr[i]; e < row_ptr[i + 1]; ++e) {
        const int j = col[e];
        float delta[DIM];
        for (int d = 0; d < DIM; ++d) delta[d] = mine[d] - pos[j * DIM + d];
        float length_sq = delta[0] * delta[0];
        for (int d = 1; d < DIM; ++d) length_sq += delta[d] * delta[d];
        const float length = sqrtf(length_sq) + 1e-9f;
        const float f = neg_k * length;
        for (int d = 0; d < DIM; ++d) total[d] += f * (delta[d] / length);
    }
    for (int d = 0; d < DIM; ++d) acc[i * DIM + d] += total[d];
}
"""

KERNEL_NAMES = ("ssn_repulsion", "ssn_springs")


# ---------------------------------------------------------------------------
# The same formulas in PyTorch: fallback path and self-check reference
# ---------------------------------------------------------------------------

def _force_params(params):
    return (
        float(params.get("COULOMB_K", 50.0)),
        float(params.get("MAX_FORCE_LIMIT", 20.0)),
        float(params.get("MAX_TOTAL_REPULSION_FORCE", 0.0)),
        float(params.get("COULOMB_CUTOFF", 15.0)),
    )


def chunk_rows(node_count, pairs=CHUNK_PAIRS):
    """Rows per block so that one block evaluates at most `pairs` pairs."""
    return max(1, min(max(node_count, 1), pairs // max(node_count, 1)))


def torch_repulsion(pos, comp, active, params, *, rows=None):
    """All-pairs repulsion with the original SSNSimulationGPU formulas.

    Rows are evaluated `rows` at a time (default: `chunk_rows`), so memory stays
    bounded; each row's sum is the same as the all-at-once evaluation. Works on
    any torch device and in float64.
    """
    k_coul, max_f, max_total, cutoff = _force_params(params)
    node_count = pos.shape[0]
    out = torch.zeros_like(pos)
    if node_count == 0:
        return out
    rows = chunk_rows(node_count) if rows is None else max(1, int(rows))
    taper_start = cutoff * 0.8
    taper_width = max(cutoff * 0.2, 1e-9)
    for first in range(0, node_count, rows):
        last = min(node_count, first + rows)
        delta = pos[first:last].unsqueeze(1) - pos.unsqueeze(0)
        dist_sq = (delta * delta).sum(dim=2)
        dist = torch.sqrt(dist_sq)
        direction_denominator = torch.where(dist_sq > 0.0, dist, torch.ones_like(dist))
        pair_mask = (
            active[first:last].unsqueeze(1)
            & active.unsqueeze(0)
            & (comp[first:last].unsqueeze(1) == comp.unsqueeze(0))
            & (dist_sq > 0.0)
            & (dist_sq <= cutoff * cutoff)
        )
        f_mag = k_coul / dist.clamp(min=0.5).pow(2)
        if max_f > 0.0:
            f_mag.clamp_(max=max_f)
        taper = ((cutoff - dist) / taper_width).clamp(min=0.0, max=1.0)
        taper = torch.where(dist > taper_start, taper, 1.0)
        f_mag = torch.where(pair_mask, f_mag * taper, 0.0)
        out[first:last] = (
            f_mag.unsqueeze(2) * (delta / direction_denominator.unsqueeze(2))
        ).sum(dim=1)
        del delta, dist_sq, dist, direction_denominator, pair_mask, f_mag, taper
    return _cap_total_repulsion(out, max_total)


def _cap_total_repulsion(repulsion, max_total):
    if max_total > 0.0:
        norm = repulsion.norm(dim=1, keepdim=True)
        repulsion *= (max_total / norm.clamp(min=1e-12)).clamp(max=1.0)
    return repulsion


def _active_springs(edges, active):
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(edges):
        return edges
    return edges[active[edges[:, 0]] & active[edges[:, 1]]]


class TorchRepulsion:
    """Fallback repulsion: `torch_repulsion` on the simulation's own tensors."""

    uses_kernels = False

    def __init__(self, comp, active):
        self.comp = comp
        self.active = active

    def compute(self, pos, params):
        return torch_repulsion(pos, self.comp, self.active, params)


class TorchSprings:
    """Fallback springs: SSNSimulationGPU's original per-edge formulas."""

    def __init__(self, edges, active, device):
        pairs = _active_springs(edges, active)
        self.a = torch.as_tensor(pairs[:, 0], device=device)
        self.b = torch.as_tensor(pairs[:, 1], device=device)

    def add(self, pos, acc, spring_k):
        if not len(self.a):
            return acc
        pa, pb = pos[self.a], pos[self.b]
        length = (pa - pb).norm(dim=1) + 1e-9
        force = (-spring_k * length).unsqueeze(1) * ((pa - pb) / length.unsqueeze(1))
        acc.index_add_(0, self.a, force)
        acc.index_add_(0, self.b, -force)
        return acc


# ---------------------------------------------------------------------------
# Runtime compiler and module loader (NVRTC + CUDA driver, hiprtc + HIP)
# ---------------------------------------------------------------------------

def _rocm_sdk_library(name):
    """Absolute path of a ROCm library from AMD's rocm_sdk wheels, if present."""
    try:
        import rocm_sdk  # AMD's "TheRock" Python packaging
        found = rocm_sdk.find_libraries(name)
    except Exception:
        return None
    return str(found[0]) if found else None


def _package_library_globs(patterns):
    """Library files matching `patterns` in torch/lib and site-packages/nvidia."""
    torch_root = os.path.dirname(os.path.abspath(torch.__file__))
    roots = [os.path.join(torch_root, "lib"), os.path.join(os.path.dirname(torch_root), "nvidia")]
    found = []
    for root in roots:
        for pattern in patterns:
            found.extend(sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True)))
    return found


def rtc_library_candidates(hip=None, platform=None):
    """Names or paths to try for the runtime compiler, best first."""
    hip = bool(torch.version.hip) if hip is None else hip
    platform = sys.platform if platform is None else platform
    windows = platform == "win32"
    candidates = []
    if hip:
        sdk = _rocm_sdk_library("hiprtc")
        if sdk:
            candidates.append(sdk)
        if windows:
            parts = str(torch.version.hip or "").split(".")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                candidates.append(f"hiprtc{int(parts[0]):02d}{int(parts[1]):02d}.dll")
            candidates += _package_library_globs(["hiprtc*.dll"])
        else:
            candidates += ["libhiprtc.so"] + _package_library_globs(["libhiprtc.so*"])
    else:
        major = str(torch.version.cuda or "0").split(".")[0]
        if windows:
            candidates.append(f"nvrtc64_{major}0_0.dll")
            candidates += _package_library_globs([f"nvrtc64_{major}*.dll", "nvrtc64_*.dll"])
        else:
            candidates += [f"libnvrtc.so.{major}", "libnvrtc.so"]
            candidates += _package_library_globs([f"libnvrtc.so.{major}*", "libnvrtc.so*"])
    return list(dict.fromkeys(candidates))


def driver_library_candidates(hip=None, platform=None):
    """Names or paths to try for the GPU driver/runtime module API, best first."""
    hip = bool(torch.version.hip) if hip is None else hip
    platform = sys.platform if platform is None else platform
    windows = platform == "win32"
    if not hip:
        return ["nvcuda.dll"] if windows else ["libcuda.so.1", "libcuda.so"]
    candidates = []
    sdk = _rocm_sdk_library("amdhip64")
    if sdk:
        candidates.append(sdk)
    major = str(torch.version.hip or "").split(".")[0]
    if windows:
        if major.isdigit():
            candidates.append(f"amdhip64_{major}.dll")
        candidates += ["amdhip64.dll"] + _package_library_globs(["amdhip64*.dll"])
    else:
        candidates += ["libamdhip64.so"] + _package_library_globs(["libamdhip64.so*"])
    return list(dict.fromkeys(candidates))


def _load_first(candidates, what):
    errors = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    raise OSError(f"no {what} library could be loaded ({'; '.join(errors) or 'no candidates'})")


def compile_options(dims, *, hip=None, arch=None):
    """Compiler options for `KERNEL_SOURCE` on one device architecture.

    NVIDIA: --gpu-architecture=sm_XY (or compute_XY for a PTX fallback) and
    --fmad=false. ROCm: --offload-arch=<gcnArchName> and -ffp-contract=off. Both
    keep each product separately rounded, as PyTorch's own per-op kernels do.
    """
    hip = bool(torch.version.hip) if hip is None else hip
    options = [f"-DDIM={int(dims)}"]
    if hip:
        options += [f"--offload-arch={arch}", "-ffp-contract=off"]
    else:
        options += [f"--gpu-architecture={arch}", "--fmad=false"]
    return options


class RuntimeCompiler:
    """NVRTC or hiprtc plus the module-launch API, bound through ctypes."""

    def __init__(self, hip=None):
        self.hip = bool(torch.version.hip) if hip is None else hip
        self.rtc, self.rtc_path = _load_first(rtc_library_candidates(self.hip), "runtime compiler")
        self.driver, self.driver_path = _load_first(driver_library_candidates(self.hip), "GPU driver")
        rtc = "hiprtc" if self.hip else "nvrtc"
        api = "hip" if self.hip else "cu"
        self._rtc = {
            name: getattr(self.rtc, f"{rtc}{name}")
            for name in ("CreateProgram", "CompileProgram", "GetProgramLogSize", "GetProgramLog",
                         "DestroyProgram", "GetErrorString")
        }
        if self.hip:
            self._rtc["GetCodeSize"] = self.rtc.hiprtcGetCodeSize
            self._rtc["GetCode"] = self.rtc.hiprtcGetCode
        else:
            self._rtc["GetCodeSize"] = self.rtc.nvrtcGetCUBINSize
            self._rtc["GetCode"] = self.rtc.nvrtcGetCUBIN
            self._rtc["GetPTXSize"] = self.rtc.nvrtcGetPTXSize
            self._rtc["GetPTX"] = self.rtc.nvrtcGetPTX
        self._rtc["GetErrorString"].restype = ctypes.c_char_p
        self._module_load = getattr(self.driver, f"{api}ModuleLoadData")
        self._get_function = getattr(self.driver, f"{api}ModuleGetFunction")
        self._launch = getattr(self.driver, "hipModuleLaunchKernel" if self.hip else "cuLaunchKernel")
        self._launch.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        if self.hip:
            self.driver.hipGetErrorString.restype = ctypes.c_char_p

    @property
    def name(self):
        return "hiprtc" if self.hip else "NVRTC"

    def version(self):
        major, minor = ctypes.c_int(), ctypes.c_int()
        getter = getattr(self.rtc, "hiprtcVersion" if self.hip else "nvrtcVersion", None)
        if getter is None or getter(ctypes.byref(major), ctypes.byref(minor)) != 0:
            return None
        return f"{major.value}.{minor.value}"

    def supported_cuda_archs(self):
        """NVRTC's supported compute capabilities (e.g. 75, 89), newest last."""
        count = ctypes.c_int()
        if self.hip or self.rtc.nvrtcGetNumSupportedArchs(ctypes.byref(count)) != 0:
            return []
        archs = (ctypes.c_int * count.value)()
        if self.rtc.nvrtcGetSupportedArchs(archs) != 0:
            return []
        return sorted(int(value) for value in archs)

    def _rtc_error(self, code):
        message = self._rtc["GetErrorString"](code)
        return message.decode(errors="replace") if message else f"error {code}"

    def _driver_check(self, code, what):
        if code == 0:
            return
        if self.hip:
            message = self.driver.hipGetErrorString(code)
        else:
            text = ctypes.c_char_p()
            self.driver.cuGetErrorString(code, ctypes.byref(text))
            message = text.value
        detail = message.decode(errors="replace") if message else f"error {code}"
        raise RuntimeError(f"{what} failed: {detail}")

    def compile(self, source, options):
        """Compile `source`; return (binary image, compiler log)."""
        program = ctypes.c_void_p()
        code = self._rtc["CreateProgram"](
            ctypes.byref(program), source.encode(), b"ssn_layout_kernels.cu", 0, None, None)
        if code != 0:
            raise RuntimeError(f"{self.name} could not create the program: {self._rtc_error(code)}")
        try:
            encoded = [option.encode() for option in options]
            array = (ctypes.c_char_p * len(encoded))(*encoded)
            result = self._rtc["CompileProgram"](program, len(encoded), array)
            log_size = ctypes.c_size_t()
            self._rtc["GetProgramLogSize"](program, ctypes.byref(log_size))
            log = ctypes.create_string_buffer(max(log_size.value, 1))
            self._rtc["GetProgramLog"](program, log)
            log_text = log.value.decode(errors="replace").strip()
            if result != 0:
                raise RuntimeError(
                    f"{self.name} compilation failed ({self._rtc_error(result)}): {log_text}")
            size = ctypes.c_size_t()
            if self._rtc["GetCodeSize"](program, ctypes.byref(size)) == 0 and size.value:
                image = ctypes.create_string_buffer(size.value)
                self._rtc["GetCode"](program, image)
                return image.raw, log_text
            if self.hip:
                raise RuntimeError("hiprtc produced no code object")
            # A compute_XY (virtual) target yields PTX only; the driver compiles it.
            self._rtc["GetPTXSize"](program, ctypes.byref(size))
            image = ctypes.create_string_buffer(size.value)
            self._rtc["GetPTX"](program, image)
            return image.raw, log_text
        finally:
            self._rtc["DestroyProgram"](ctypes.byref(program))

    def load(self, image, names):
        """Load a compiled image into the current device context."""
        module = ctypes.c_void_p()
        self._driver_check(self._module_load(ctypes.byref(module), image), "loading the kernel module")
        functions = {}
        for name in names:
            function = ctypes.c_void_p()
            self._driver_check(
                self._get_function(ctypes.byref(function), module, name.encode()),
                f"finding kernel {name}")
            functions[name] = function
        return module, functions

    def launch(self, function, grid, block, args, stream):
        holders = []
        pointers = (ctypes.c_void_p * len(args))()
        for index, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                value = ctypes.c_void_p(arg.data_ptr())
            elif isinstance(arg, float):
                value = ctypes.c_double(arg)
            else:
                value = ctypes.c_int(int(arg))
            holders.append(value)
            pointers[index] = ctypes.addressof(value)
        self._driver_check(
            self._launch(function, *grid, *block, 0, ctypes.c_void_p(stream), pointers, None),
            "launching a layout kernel")


# ---------------------------------------------------------------------------
# Compiled kernels per device, and the per-simulation plans that use them
# ---------------------------------------------------------------------------

_COMPILER = None
_COMPILER_ERROR = None
_LOCK = threading.RLock()
_KERNELS = {}
_REASONS = {}


def _compiler():
    global _COMPILER, _COMPILER_ERROR
    if _COMPILER is None and _COMPILER_ERROR is None:
        try:
            _COMPILER = RuntimeCompiler()
        except Exception as error:
            _COMPILER_ERROR = f"{type(error).__name__}: {error}"
    if _COMPILER is None:
        raise RuntimeError(_COMPILER_ERROR)
    return _COMPILER


def _device_index(device):
    return torch.cuda.current_device() if device.index is None else int(device.index)


def kernels_disabled():
    value = os.environ.get(DISABLE_ENVIRONMENT_VARIABLE, "").strip().lower()
    return value in _OFF_VALUES


def kernel_cache_folder(platform=None):
    """The per-user folder for compiled kernels, or None when caching is off."""
    configured = os.environ.get(CACHE_ENVIRONMENT_VARIABLE)
    if configured is not None and configured.strip():
        return None if configured.strip().lower() in _OFF_VALUES else configured
    platform = sys.platform if platform is None else platform
    home = os.path.expanduser("~")
    if platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return os.path.join(base, "EMAP-SSN", "gpu_kernels")
    if platform == "darwin":
        return os.path.join(home, "Library", "Caches", "EMAP-SSN", "gpu_kernels")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(home, ".cache")
    return os.path.join(base, "emap-ssn", "gpu_kernels")


def _cache_path(compiler, arch, dims):
    """Cache file for one compiler, architecture and dimensionality, or None."""
    folder = kernel_cache_folder()
    if not folder:
        return None
    material = "\n".join([
        KERNEL_SOURCE, " ".join(compile_options(dims, hip=compiler.hip, arch=arch)),
        compiler.name, compiler.version() or "?", str(torch.version.hip or torch.version.cuda),
    ])
    digest = hashlib.sha256(material.encode()).hexdigest()[:24]
    vendor = "hip" if compiler.hip else "cuda"
    safe_arch = re.sub(r"[^A-Za-z0-9_.-]", "_", arch)
    return os.path.join(folder, f"ssn-layout-{vendor}-{safe_arch}-{int(dims)}d-{digest}.bin")


class LayoutKernels:
    """The two layout kernels compiled and loaded for one device and dimensionality.

    A binary from the per-user cache is loaded instead of compiling when one
    exists; `store_in_cache` keeps a freshly compiled binary once it has passed
    the self-check.
    """

    def __init__(self, device, dims, *, read_cache=True):
        self.device = torch.device("cuda", _device_index(device))
        self.dims = int(dims)
        self.compiler = _compiler()
        properties = torch.cuda.get_device_properties(self.device)
        self.device_name = properties.name
        self.multiprocessors = max(1, int(properties.multi_processor_count))
        self.arch = (str(properties.gcnArchName) if self.compiler.hip
                     else f"sm_{properties.major}{properties.minor}")
        self.cache_path = _cache_path(self.compiler, self.arch, self.dims)
        self.from_cache = False
        self._image = None
        with torch.cuda.device(self.device):
            torch.empty(1, device=self.device)   # the context must exist before the driver API
            if read_cache and self.cache_path:
                self._load_cached()
            if not self.from_cache:
                if self.compiler.hip:
                    self.target, self._image = self.arch, self.compiler.compile(
                        KERNEL_SOURCE, compile_options(self.dims, hip=True, arch=self.arch))[0]
                else:
                    self.target, self._image = self._compile_cuda(properties)
                self._module, self._functions = self.compiler.load(self._image, KERNEL_NAMES)

    def _load_cached(self):
        try:
            with open(self.cache_path, "rb") as handle:
                target, separator, image = handle.read().partition(b"\n")
        except OSError:
            return
        try:
            if not separator or not image:
                raise RuntimeError("truncated cache file")
            self._module, self._functions = self.compiler.load(image, KERNEL_NAMES)
        except RuntimeError:
            self.discard_cache()
            return
        self.target = target.decode(errors="replace")
        self.from_cache = True

    def store_in_cache(self):
        """Keep a freshly compiled binary for later processes (best effort)."""
        if not self.cache_path or self.from_cache or self._image is None:
            return
        folder = os.path.dirname(self.cache_path)
        temporary = None
        try:
            os.makedirs(folder, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=folder, suffix=".tmp")
            with os.fdopen(handle, "wb") as stream:
                stream.write(self.target.encode() + b"\n" + self._image)
            os.replace(temporary, self.cache_path)
        except OSError:
            if temporary:
                try:
                    os.remove(temporary)
                except OSError:
                    pass

    def discard_cache(self):
        if self.cache_path:
            try:
                os.remove(self.cache_path)
            except OSError:
                pass

    def _compile_cuda(self, properties):
        capability = properties.major * 10 + properties.minor
        try:
            target = f"sm_{capability}"
            image, _ = self.compiler.compile(
                KERNEL_SOURCE, compile_options(self.dims, hip=False, arch=target))
            return target, image
        except RuntimeError as native_error:
            # A GPU newer than this NVRTC: emit PTX for the newest architecture it
            # knows and let the driver compile it for the device.
            known = [arch for arch in self.compiler.supported_cuda_archs() if arch <= capability]
            if not known:
                raise native_error
            target = f"compute_{known[-1]}"
            image, _ = self.compiler.compile(
                KERNEL_SOURCE, compile_options(self.dims, hip=False, arch=target))
            return target, image

    def describe(self):
        version = self.compiler.version()
        compiler = f"{self.compiler.name} {version}" if version else self.compiler.name
        source = ", cached" if self.from_cache else ""
        return f"runtime-compiled kernels ({compiler}, {self.target}{source})"

    def launch(self, name, grid, block, args):
        stream = torch.cuda.current_stream(self.device).cuda_stream
        with torch.cuda.device(self.device):
            self.compiler.launch(self._functions[name], grid, block, args, stream)

    def repulsion_plan(self, comp_labels, active, node_count, **options):
        return KernelRepulsion(self, comp_labels, active, node_count, **options)

    def spring_plan(self, edges, active, node_count):
        return KernelSprings(self, edges, active, node_count)

    def self_check(self):
        """Compare both kernels with float64 PyTorch on two fixed small problems."""
        cpu = torch.device("cpu")
        cases = [(_self_check_problem(self.dims), (0.0, 3.0), (False, True), (1, 3)),
                 (_tile_gap_problem(self.dims), (0.0,), (True,), (1, 2))]
        for problem, caps, skips, splits in cases:
            pos64 = torch.tensor(problem["pos"], dtype=torch.float64, device=cpu)
            comp_cpu = torch.tensor(problem["comp"], device=cpu)
            active_cpu = torch.tensor(problem["active"], device=cpu)
            pos = torch.tensor(problem["pos"], device=self.device)
            count = len(problem["pos"])
            for max_total in caps:
                params = dict(problem["params"], MAX_TOTAL_REPULSION_FORCE=max_total)
                expected = torch_repulsion(pos64, comp_cpu, active_cpu, params)
                for use_skip in skips:
                    for j_split in splits:
                        plan = self.repulsion_plan(problem["comp"], problem["active"], count,
                                                   use_skip=use_skip, j_split=j_split)
                        if use_skip and not plan.skipped_tiles(pos, params):
                            raise RuntimeError(f"self-check problem {problem['name']} skips no tiles")
                        got = plan.compute(pos, params).double().cpu()
                        _assert_close(got, expected,
                                      f"{problem['name']} repulsion (skip={use_skip}, "
                                      f"split={j_split}, total cap={max_total})")
        problem = cases[0][0]
        pos64 = torch.tensor(problem["pos"], dtype=torch.float64, device=cpu)
        springs_expected = TorchSprings(problem["edges"], problem["active"], cpu).add(
            pos64, torch.zeros_like(pos64), problem["params"]["SPRING_K"])
        springs = self.spring_plan(problem["edges"], problem["active"], len(problem["pos"]))
        pos = torch.tensor(problem["pos"], device=self.device)
        got = springs.add(pos, torch.zeros_like(pos), problem["params"]["SPRING_K"])
        _assert_close(got.double().cpu(), springs_expected, "springs")


def _assert_close(got, expected, what):
    scale = float(expected.abs().max()) if expected.numel() else 0.0
    error = float((got - expected).abs().max()) if expected.numel() else 0.0
    if not math.isfinite(error) or error > _SELF_CHECK_TOLERANCE * max(scale, 1.0):
        raise RuntimeError(
            f"self-check failed for {what}: max error {error:.3g} against max force {scale:.3g}")


def _self_check_problem(dims):
    """A fixed problem with capped, tapered, cut-off and cross-component pairs.

    Component 0 is tight (pairs closer than the 0.71 at which COULOMB_K hits
    MAX_FORCE_LIMIT), component 1 spans the taper zone, and component 2 is two
    blobs 200 apart, so once sorted some of its tiles lie beyond the cutoff of
    each other. A tenth of the nodes are inactive, and the node order
    interleaves the components.
    """
    rng = np.random.default_rng(42)
    offset = np.zeros(dims)
    offset[0] = 100.0
    blocks = [rng.normal(0.0, 0.35, (220, dims)), rng.normal(0.0, 6.0, (180, dims)),
              rng.normal(0.0, 8.0, (180, dims)) - offset, rng.normal(0.0, 8.0, (180, dims)) + offset]
    sizes = (220, 180, 360)
    pos = np.concatenate(blocks).astype(np.float32)
    comp = np.repeat(np.arange(len(sizes)), sizes).astype(np.int64)
    order = rng.permutation(len(pos))
    pos, comp = pos[order], comp[order]
    active = rng.random(len(pos)) > 0.1
    edges = []
    for label in range(len(sizes)):
        members = np.flatnonzero(comp == label)
        edges.append(np.column_stack((rng.choice(members, 6 * len(members)),
                                      rng.choice(members, 6 * len(members)))))
    params = {"COULOMB_K": 10.0, "MAX_FORCE_LIMIT": 20.0, "COULOMB_CUTOFF": 15.0,
              "SPRING_K": 5.0}
    return {"name": "mixed", "pos": pos, "comp": comp, "active": active,
            "edges": np.concatenate(edges).astype(np.int64), "params": params}


def _tile_gap_problem(dims):
    """One component of four 128-node blobs on a line, one tile each in run order.

    Blobs 0 and 1 (and 2 and 3) are 12 apart: within the cutoff of each other
    although their tile boxes do not touch. The pairs 200 apart must be skipped,
    and nothing else may be. The blobs are far apart along the first axis only,
    so the Morton sort keeps each one contiguous.
    """
    rng = np.random.default_rng(42)
    blocks = []
    for centre in (0.0, 12.0, 200.0, 212.0):
        block = rng.normal(0.0, 0.5, (TILE, dims))
        block[:, 0] += centre
        blocks.append(block)
    count = TILE * 4
    params = {"COULOMB_K": 10.0, "MAX_FORCE_LIMIT": 20.0, "COULOMB_CUTOFF": 15.0}
    return {"name": "tile-gap", "pos": np.concatenate(blocks).astype(np.float32),
            "comp": np.zeros(count, np.int64), "active": np.ones(count, bool),
            "edges": np.zeros((0, 2), np.int64), "params": params}


def _spread_bits(values, dims):
    """Interleave zeros between the low bits of `values` (Morton encoding)."""
    if dims == 2:
        values = (values | (values << 8)) & 0x00FF00FF
        values = (values | (values << 4)) & 0x0F0F0F0F
        values = (values | (values << 2)) & 0x33333333
        return (values | (values << 1)) & 0x55555555
    values = (values | (values << 16)) & 0x030000FF
    values = (values | (values << 8)) & 0x0300F00F
    values = (values | (values << 4)) & 0x030C30C3
    return (values | (values << 2)) & 0x09249249


class KernelRepulsion:
    """Per-simulation state of the repulsion kernel.

    Active nodes are grouped by component (the CPU kernel's run order), so a
    node's partners are one contiguous run. For large components the run is
    reordered along a Morton curve every RESORT_INTERVAL steps, and tiles whose
    bounding boxes, rebuilt every step, lie beyond the cutoff are skipped; the
    order only decides how much is skipped, never the result's correctness.
    """

    uses_kernels = True

    def __init__(self, kernels, comp_labels, active, node_count, *, use_skip=None, j_split=None):
        self.kernels = kernels
        self.device = kernels.device
        self.dims = kernels.dims
        self.node_count = int(node_count)
        comp = np.asarray(comp_labels, dtype=np.int64).reshape(-1)
        active = np.asarray(active, dtype=bool).reshape(-1)
        nodes = np.flatnonzero(active)
        members = nodes[np.argsort(comp[nodes], kind="mergesort")]
        member_comp = comp[members]
        starts = np.zeros(len(members), np.int32)
        ends = np.zeros(len(members), np.int32)
        if len(members):
            bounds = np.concatenate(([0], np.flatnonzero(np.diff(member_comp)) + 1, [len(members)]))
            for low, high in zip(bounds[:-1], bounds[1:]):
                starts[low:high], ends[low:high] = low, high
        self.run_count = len(members)
        self.tile_count = (self.run_count + TILE - 1) // TILE
        self.run_node = torch.tensor(members.astype(np.int32), device=self.device)
        self.run_index = self.run_node.long()
        self.run_start = torch.tensor(starts, device=self.device)
        self.run_end = torch.tensor(ends, device=self.device)
        self.member_comp = torch.tensor(member_comp, device=self.device)
        self.use_skip = (self.run_count >= SKIP_MIN_NODES) if use_skip is None else bool(use_skip)
        if j_split is None:
            target = BLOCKS_PER_MULTIPROCESSOR * kernels.multiprocessors
            j_split = -(-target // max(self.tile_count, 1))
        self.j_split = int(max(1, min(j_split, max(self.tile_count, 1))))
        self.tile_box = torch.zeros((max(self.tile_count, 1), 2 * self.dims),
                                    dtype=torch.float32, device=self.device)
        self.partial = (
            torch.zeros((self.j_split, self.node_count, self.dims), dtype=torch.float32, device=self.device)
            if self.j_split > 1 else torch.zeros(1, device=self.device))
        self._steps_since_sort = None

    def _sort(self, pos):
        cell = float(self._cutoff) / (8.0 if self.dims == 2 else 4.0)
        points = pos.index_select(0, self.run_index)
        limit = 65535 if self.dims == 2 else 1023
        cells = torch.floor((points - points.min(dim=0).values) / cell).long().clamp(0, limit)
        key = torch.zeros_like(cells[:, 0])
        for axis in range(self.dims):
            key |= _spread_bits(cells[:, axis], self.dims) << axis
        order = torch.argsort(self.member_comp * (1 << 32) + key, stable=True)
        self.run_node = self.run_node[order].contiguous()
        self.run_index = self.run_node.long()

    def _tile_boxes(self, run_pos):
        padding = self.tile_count * TILE - self.run_count
        if padding:
            run_pos = torch.cat((run_pos, run_pos[-1:].expand(padding, self.dims)))
        tiles = run_pos.view(self.tile_count, TILE, self.dims)
        self.tile_box = torch.cat((tiles.amin(dim=1), tiles.amax(dim=1)), dim=1).contiguous()

    def skipped_tiles(self, pos, params):
        """How many tile pairs the next compute() may skip (for checks and logs)."""
        self._cutoff = _force_params(params)[3]
        if not self.use_skip or not self.run_count:
            return 0
        self._sort(pos)
        self._steps_since_sort = 0
        self._tile_boxes(pos.index_select(0, self.run_index))
        boxes = self.tile_box
        low, high = boxes[:, :self.dims], boxes[:, self.dims:]
        gap = torch.maximum(low.unsqueeze(1) - high.unsqueeze(0), low.unsqueeze(0) - high.unsqueeze(1))
        gap_sq = gap.clamp(min=0.0).pow(2).sum(dim=2)
        return int((gap_sq > self._cutoff * self._cutoff * 1.0001 + 1e-6).sum())

    def compute(self, pos, params):
        k_coul, max_f, max_total, cutoff = _force_params(params)
        self._cutoff = cutoff
        if not self.run_count:
            return torch.zeros((self.node_count, self.dims), dtype=torch.float32, device=self.device)
        # Split shares write every active row of `partial`; otherwise the kernel
        # writes the active rows of `out` and inactive rows stay zero.
        out = self.partial if self.j_split > 1 else torch.zeros(
            (self.node_count, self.dims), dtype=torch.float32, device=self.device)
        if self.use_skip:
            if self._steps_since_sort is None or self._steps_since_sort >= RESORT_INTERVAL:
                self._sort(pos)
                self._steps_since_sort = 0
            self._steps_since_sort += 1
        run_pos = pos.index_select(0, self.run_index).contiguous()
        if self.use_skip:
            self._tile_boxes(run_pos)
        self.kernels.launch(
            "ssn_repulsion", (self.tile_count, self.j_split, 1), (TILE, 1, 1),
            [run_pos, self.run_node, self.run_start, self.run_end, self.tile_box, out,
             self.partial, self.node_count, self.run_count, self.j_split, int(self.use_skip),
             k_coul, max_f, max_total, cutoff])
        if self.j_split > 1:
            out = _cap_total_repulsion(self.partial.sum(dim=0), max_total)
        return out


class KernelSprings:
    """Per-simulation symmetric CSR list of active springs for the spring kernel."""

    def __init__(self, kernels, edges, active, node_count):
        self.kernels = kernels
        self.node_count = int(node_count)
        pairs = torch.as_tensor(_active_springs(edges, np.asarray(active, dtype=bool)),
                                device=kernels.device)
        self.spring_count = int(pairs.shape[0])
        rows = torch.cat((pairs[:, 0], pairs[:, 1]))
        columns = torch.cat((pairs[:, 1], pairs[:, 0]))
        del pairs
        order = torch.argsort(rows, stable=True)
        self.col = columns[order].to(torch.int32).contiguous()
        del columns, order
        self.row_ptr = torch.zeros(self.node_count + 1, dtype=torch.int64, device=kernels.device)
        self.row_ptr[1:] = torch.cumsum(torch.bincount(rows, minlength=self.node_count), dim=0)

    def add(self, pos, acc, spring_k):
        if self.spring_count and self.node_count:
            self.kernels.launch(
                "ssn_springs", ((self.node_count + SPRING_BLOCK - 1) // SPRING_BLOCK, 1, 1),
                (SPRING_BLOCK, 1, 1),
                [pos, self.row_ptr, self.col, acc, self.node_count, float(spring_k)])
        return acc


def layout_kernels(device, dims):
    """Validated kernels for a CUDA or ROCm torch device, or None.

    None means the PyTorch fallback must be used; `fallback_reason` says why.
    Compilation and the self-check run once per device and dimensionality.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    if kernels_disabled():
        return None
    key = (_device_index(device), int(dims))
    with _LOCK:
        if key not in _KERNELS:
            try:
                kernels = LayoutKernels(device, dims)
                try:
                    kernels.self_check()
                except Exception:
                    if not kernels.from_cache:
                        raise
                    # A cached binary that disagrees is dropped and rebuilt once.
                    kernels.discard_cache()
                    kernels = LayoutKernels(device, dims, read_cache=False)
                    kernels.self_check()
                kernels.store_in_cache()
                _KERNELS[key] = kernels
            except Exception as error:
                _KERNELS[key] = None
                _REASONS[key] = f"{type(error).__name__}: {error}"
                name = _safe_device_name(key[0])
                print(f"GPU layout kernels are unavailable on {name} ({_REASONS[key]}); "
                      "using the PyTorch fallback, which is slower and needs memory "
                      "for every node pair.")
        return _KERNELS[key]


def _safe_device_name(index):
    try:
        return torch.cuda.get_device_name(index)
    except Exception:
        return f"cuda:{index}"


def fallback_reason(device, dims):
    """Why `layout_kernels` returns None for this device, or None if it doesn't."""
    device = torch.device(device)
    if device.type != "cuda":
        return f"no runtime kernels for {device.type} devices"
    if kernels_disabled():
        return f"disabled by {DISABLE_ENVIRONMENT_VARIABLE}"
    return _REASONS.get((_device_index(device), int(dims)))


def describe(device, dims):
    """One line saying which implementation SSNSimulationGPU uses on `device`."""
    kernels = layout_kernels(device, dims)
    if kernels is not None:
        return kernels.describe()
    return f"PyTorch fallback ({fallback_reason(device, dims) or 'kernels unavailable'})"


__all__ = [
    "CACHE_ENVIRONMENT_VARIABLE",
    "CHUNK_PAIRS",
    "DISABLE_ENVIRONMENT_VARIABLE",
    "KERNEL_SOURCE",
    "LayoutKernels",
    "RuntimeCompiler",
    "TorchRepulsion",
    "TorchSprings",
    "chunk_rows",
    "compile_options",
    "describe",
    "driver_library_candidates",
    "fallback_reason",
    "kernel_cache_folder",
    "kernels_disabled",
    "layout_kernels",
    "rtc_library_candidates",
    "torch_repulsion",
]
