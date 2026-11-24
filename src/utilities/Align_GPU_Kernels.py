# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Runtime-compiled GPU kernels for fused residue-embedding alignment.

Computes pairwise similarity transformation, exact two-pass population Z-score
normalization, and exact linear-gap Global (Needleman-Wunsch) + Local
(Smith-Waterman) dynamic programming directly on GPU-resident cosine matrices
produced by dense block-GEMM (or zero-padding row-packed GEMM).

This eliminates:
  1. Per-microbatch zero-padding and target gathering (`_gather_padded_targets`)
  2. ~30 intermediate VRAM passes in eager PyTorch Z-score normalization
  3. PCIe device-to-host transfer of full L_i x L_j score matrices (~360 KB/pair
     reduced to 16 bytes/pair, a >20,000x transfer reduction)
  4. CPU DP bottlenecks and pinned host score-matrix ring buffers.

Compiled with NVRTC (NVIDIA) or hiprtc (AMD ROCm) using the same zero-header,
zero-toolkit runtime compiler infrastructure as `utilities.Layout_GPU_Kernels`,
with `--fmad=false` / `-ffp-contract=off`, per-user binary caching, and an
automatic self-check against CPU reference kernels before first use.
Setting `SSN_ALIGN_GPU_KERNELS=0` disables these kernels and falls back cleanly
to the PyTorch tiled pipeline.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import threading
from collections import OrderedDict, deque
from contextlib import ExitStack, nullcontext

import h5py
import numpy as np
import torch

from utilities import Layout_GPU_Kernels as _gpu_rt
from utilities.Network_Kernels import (
    PACKED_MAX_PAIRS,
    PACKED_MAX_PATH,
    float32_exact_local_bound,
    global_local_scores,
    global_score_length_identity,
    local_score_length_identity,
)


DISABLE_ENVIRONMENT_VARIABLE = "SSN_ALIGN_GPU_KERNELS"
_OFF_VALUES = {"0", "off", "false", "no"}
# Maximum allowable per-residue score difference in the startup self-check.
_SELF_CHECK_PER_RESIDUE_TOL = 1e-4
# Block-GEMM target residue span (scales down automatically on low-VRAM GPUs).
DEFAULT_BLOCK_RESIDUES = 8192
MIN_BLOCK_RESIDUES = 1024
# Use 2D dense block-GEMM when requested pair cell density >= this threshold;
# otherwise pack each row's exact targets contiguously with zero padding.
DENSE_BLOCK_DENSITY_THRESHOLD = 0.20


KERNEL_SOURCE = r"""
#define STAT_BLOCK 128
#define DP_WARPS_PER_BLK 4

#ifdef __HIP_PLATFORM_AMD__
/* Portable 32-lane virtual warp helpers using shared memory so kernels work
   identically on wave32 and wave64 AMD GPUs with zero external headers. */
#define SYNC_WARP() __syncthreads()
#else
#define SYNC_WARP() __syncwarp()
#endif

static __device__ inline float warp_sum_32(float v, volatile float* smem_warp, int lane) {
#ifdef __HIP_PLATFORM_AMD__
    smem_warp[lane] = v;
    __syncthreads();
    if (lane < 16) smem_warp[lane] += smem_warp[lane + 16];
    __syncthreads();
    if (lane < 8)  smem_warp[lane] += smem_warp[lane + 8];
    __syncthreads();
    if (lane < 4)  smem_warp[lane] += smem_warp[lane + 4];
    __syncthreads();
    if (lane < 2)  smem_warp[lane] += smem_warp[lane + 2];
    __syncthreads();
    if (lane == 0) smem_warp[0] += smem_warp[1];
    __syncthreads();
    return smem_warp[0];
#else
    (void)smem_warp;
    (void)lane;
    v += __shfl_down_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 8);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 4);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 2);
    v += __shfl_down_sync(0xFFFFFFFFu, v, 1);
    return __shfl_sync(0xFFFFFFFFu, v, 0);
#endif
}

/* Kernel 1: In-place cosine -> similarity = exp(-(1 - clamp(cos, -1, 1))) and
   exact two-pass population mean and half_inv_std = 0.5 / (std + 1e-8) for
   every pair's rows and columns. Flags any non-finite input in nonfinite_flag.
   Grid: (num_pairs, 1, 1), Block: (128, 1, 1). */
extern "C" __global__ void align_exp_stats(
    float* __restrict__ mat,
    int mat_stride,
    const int* __restrict__ pair_row_off,
    const int* __restrict__ pair_row_len,
    const int* __restrict__ pair_col_off,
    const int* __restrict__ pair_col_len,
    const int* __restrict__ pair_rstat_off,
    const int* __restrict__ pair_cstat_off,
    float* __restrict__ row_mean_out,
    float* __restrict__ row_his_out,
    float* __restrict__ col_mean_out,
    float* __restrict__ col_his_out,
    int* __restrict__ nonfinite_flag,
    int num_pairs)
{
    const int p = blockIdx.x;
    if (p >= num_pairs) return;

    const int r0 = pair_row_off[p];
    const int R  = pair_row_len[p];
    const int c0 = pair_col_off[p];
    const int C  = pair_col_len[p];
    const int rs_off = pair_rstat_off[p];
    const int cs_off = pair_cstat_off[p];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ float s_red[4][32];
    int local_bad = 0;

    const float fC = (float)C;
    const int row_iters = (R + 3) >> 2;
    for (int iter = 0; iter < row_iters; ++iter) {
        const int r = iter * 4 + warp_id;
        float sum = 0.0f;
        float* row_ptr = 0;
        if (r < R) {
            row_ptr = mat + (long long)(r0 + r) * mat_stride + c0;
            for (int c = lane; c < C; c += 32) {
                float v = row_ptr[c];
                if (!(v == v) || (v - v != 0.0f)) local_bad = 1;
                float clamped = fminf(fmaxf(v, -1.0f), 1.0f);
                float s = expf(-(1.0f - clamped));
                row_ptr[c] = s;
                sum += s;
            }
        }
        const float mean = warp_sum_32(sum, s_red[warp_id], lane) / fC;

        float var_sum = 0.0f;
        if (r < R) {
            for (int c = lane; c < C; c += 32) {
                float d = row_ptr[c] - mean;
                var_sum += d * d;
            }
        }
        const float vtot = warp_sum_32(var_sum, s_red[warp_id], lane);
        if (r < R && lane == 0) {
            row_mean_out[rs_off + r] = mean;
            row_his_out[rs_off + r]  = 0.5f / (sqrtf(vtot / fC) + 1e-8f);
        }
    }

    if (local_bad) nonfinite_flag[0] = 1;
    __syncthreads();

    const float fR = (float)R;
    for (int c = tid; c < C; c += STAT_BLOCK) {
        const float* col_ptr = mat + (long long)r0 * mat_stride + (c0 + c);
        float sum = 0.0f;
        for (int r = 0; r < R; ++r) {
            sum += col_ptr[(long long)r * mat_stride];
        }
        const float mean = sum / fR;
        float var_sum = 0.0f;
        for (int r = 0; r < R; ++r) {
            float d = col_ptr[(long long)r * mat_stride] - mean;
            var_sum += d * d;
        }
        col_mean_out[cs_off + c] = mean;
        col_his_out[cs_off + c]  = 0.5f / (sqrtf(var_sum / fR) + 1e-8f);
    }
}

/* Kernel 2: Division-free register-tiled 4-columns-per-thread warp-cooperative
   wavefront DP (128 columns per swath) for exact float32 global+local alignment.
   Grid: ((num_pairs + 3) / 4, 1, 1), Block: (128, 1, 1). */
extern "C" __global__ void align_dp_f32_k4(
    const float* __restrict__ mat,
    int mat_stride,
    const int* __restrict__ pair_row_off,
    const int* __restrict__ pair_row_len,
    const int* __restrict__ pair_col_off,
    const int* __restrict__ pair_col_len,
    const int* __restrict__ pair_rstat_off,
    const int* __restrict__ pair_cstat_off,
    const float* __restrict__ row_mean,
    const float* __restrict__ row_his,
    const float* __restrict__ col_mean,
    const float* __restrict__ col_his,
    float* __restrict__ scratch_g,
    unsigned int* __restrict__ scratch_gl,
    float* __restrict__ scratch_l,
    unsigned int* __restrict__ scratch_ll,
    int scratch_stride,
    float* __restrict__ out_gs,
    unsigned int* __restrict__ out_gl,
    float* __restrict__ out_ls,
    unsigned int* __restrict__ out_ll,
    int num_pairs,
    double g_gap_d,
    double l_gap_d,
    double l_shift_d)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int p = blockIdx.x * DP_WARPS_PER_BLK + warp_id;

#ifdef __HIP_PLATFORM_AMD__
    __shared__ float s_shfl_g[DP_WARPS_PER_BLK][32];
    __shared__ unsigned int s_shfl_gl[DP_WARPS_PER_BLK][32];
    __shared__ float s_shfl_l[DP_WARPS_PER_BLK][32];
    __shared__ unsigned int s_shfl_ll[DP_WARPS_PER_BLK][32];
    __shared__ unsigned long long s_shfl_pos[DP_WARPS_PER_BLK][32];
    __shared__ int s_blk_r[DP_WARPS_PER_BLK];
    __shared__ int s_blk_c[DP_WARPS_PER_BLK];
    const bool active_pair = (p < num_pairs);
#else
    if (p >= num_pairs) return;
    const bool active_pair = true;
#endif

    const float g_gap = (float)g_gap_d;
    const float l_gap = (float)l_gap_d;
    const float l_shift = (float)l_shift_d;

    const int r0 = active_pair ? pair_row_off[p] : 0;
    const int R  = active_pair ? pair_row_len[p] : 0;
    const int c0 = active_pair ? pair_col_off[p] : 0;
    const int C  = active_pair ? pair_col_len[p] : 0;
    const float* rm_ptr   = active_pair ? (row_mean + pair_rstat_off[p]) : row_mean;
    const float* rhis_ptr = active_pair ? (row_his  + pair_rstat_off[p]) : row_his;
    const float* cm_ptr   = active_pair ? (col_mean + pair_cstat_off[p]) : col_mean;
    const float* chis_ptr = active_pair ? (col_his  + pair_cstat_off[p]) : col_his;

    float* v_g = scratch_g + (long long)(active_pair ? p : 0) * scratch_stride;
    unsigned int* v_gl = scratch_gl + (long long)(active_pair ? p : 0) * scratch_stride;
    float* v_l = scratch_l + (long long)(active_pair ? p : 0) * scratch_stride;
    unsigned int* v_ll = scratch_ll + (long long)(active_pair ? p : 0) * scratch_stride;

    float best_l = 0.0f;
    unsigned int best_ll = 0u;
    unsigned long long best_pos = 0xFFFFFFFFFFFFFFFFull;
    float final_g = 0.0f;
    unsigned int final_gl = 0u;

    const int swath_w = 128;
#ifdef __HIP_PLATFORM_AMD__
    if (lane == 0) {
        s_blk_r[warp_id] = R;
        s_blk_c[warp_id] = C;
    }
    __syncthreads();
    int blk_R = s_blk_r[0];
    int blk_C = s_blk_c[0];
    for (int w = 1; w < DP_WARPS_PER_BLK; ++w) {
        if (s_blk_r[w] > blk_R) blk_R = s_blk_r[w];
        if (s_blk_c[w] > blk_C) blk_C = s_blk_c[w];
    }
    const int loop_C = blk_C;
    const int loop_steps = blk_R + 31;
#else
    const int loop_C = C;
    const int loop_steps = R + 31;
#endif

    for (int col_base = 0; col_base < loop_C; col_base += swath_w) {
        const int c_start = col_base + lane * 4;
        float cm[4], chis[4];
        float cur_g[4], cur_l[4];
        unsigned int cur_gl[4], cur_ll[4];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int c = c_start + k;
            bool valid = (c < C);
            cm[k] = valid ? cm_ptr[c] : 0.0f;
            chis[k] = valid ? chis_ptr[c] : 0.0f;
            cur_g[k] = (float)(c + 1) * g_gap;
            cur_gl[k] = (unsigned int)(c + 1);
            cur_l[k] = 0.0f;
            cur_ll[k] = 0u;
        }
        float diag0_g = (float)c_start * g_gap;
        unsigned int diag0_gl = (unsigned int)c_start;
        float diag0_l = 0.0f;
        unsigned int diag0_ll = 0u;

        float send_g = cur_g[3];
        unsigned int send_gl = cur_gl[3];
        float send_l = cur_l[3];
        unsigned int send_ll = cur_ll[3];

        const bool full_k = (c_start + 4 <= C);

        for (int t = 0; t < loop_steps; ++t) {
#ifdef __HIP_PLATFORM_AMD__
            s_shfl_g[warp_id][lane] = send_g;
            s_shfl_gl[warp_id][lane] = send_gl;
            s_shfl_l[warp_id][lane] = send_l;
            s_shfl_ll[warp_id][lane] = send_ll;
            __syncthreads();
            float lg = (lane > 0) ? s_shfl_g[warp_id][lane - 1] : 0.0f;
            unsigned int lgl = (lane > 0) ? s_shfl_gl[warp_id][lane - 1] : 0u;
            float ll = (lane > 0) ? s_shfl_l[warp_id][lane - 1] : 0.0f;
            unsigned int lll = (lane > 0) ? s_shfl_ll[warp_id][lane - 1] : 0u;
            __syncthreads();
#else
            float lg = __shfl_up_sync(0xFFFFFFFFu, send_g, 1);
            unsigned int lgl = __shfl_up_sync(0xFFFFFFFFu, send_gl, 1);
            float ll = __shfl_up_sync(0xFFFFFFFFu, send_l, 1);
            unsigned int lll = __shfl_up_sync(0xFFFFFFFFu, send_ll, 1);
#endif
            const int r = t - lane;
            if (r >= 0 && r < R && c_start < C) {
                if (lane == 0) {
                    if (col_base == 0) {
                        lg = (float)(r + 1) * g_gap;
                        lgl = (unsigned int)(r + 1);
                        ll = 0.0f;
                        lll = 0u;
                    } else {
                        lg = v_g[r];
                        lgl = v_gl[r];
                        ll = v_l[r];
                        lll = v_ll[r];
                    }
                }
                const float rm = rm_ptr[r];
                const float rhis = rhis_ptr[r];
                const float* row_ptr = mat + (long long)(r0 + r) * mat_stride + (c0 + c_start);

                float dg = diag0_g;
                unsigned int dgl = diag0_gl;
                float dl = diag0_l;
                unsigned int dll = diag0_ll;

                diag0_g = lg; diag0_gl = lgl; diag0_l = ll; diag0_ll = lll;

                if (full_k) {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const float sim = row_ptr[k];
                        const float cell = (sim - rm) * rhis + (sim - cm[k]) * chis[k];

                        const float ug = cur_g[k];
                        const unsigned int ugl = cur_gl[k];
                        const float ul = cur_l[k];
                        const unsigned int ull = cur_ll[k];

                        float bg = dg + cell;
                        unsigned int bgl = dgl + 1u;
                        float gd = ug + g_gap;
                        if (gd > bg) { bg = gd; bgl = ugl + 1u; }
                        float gi = lg + g_gap;
                        if (gi > bg) { bg = gi; bgl = lgl + 1u; }

                        float lm = dl + (cell - l_shift);
                        float bl = 0.0f;
                        unsigned int bll = 0u;
                        if (lm > 0.0f) { bl = lm; bll = dll + 1u; }
                        float ld = ul + l_gap;
                        if (ld > bl) { bl = ld; bll = ull + 1u; }
                        float li = ll + l_gap;
                        if (li > bl) { bl = li; bll = lll + 1u; }

                        if (bl > best_l) {
                            best_l = bl; best_ll = bll;
                            best_pos = ((unsigned long long)(unsigned int)r << 32) | (unsigned int)(c_start + k);
                        } else if (bl == best_l && bl > 0.0f) {
                            unsigned long long pos = ((unsigned long long)(unsigned int)r << 32) | (unsigned int)(c_start + k);
                            if (pos < best_pos) { best_ll = bll; best_pos = pos; }
                        }

                        dg = ug; dgl = ugl; dl = ul; dll = ull;
                        cur_g[k] = bg; cur_gl[k] = bgl; cur_l[k] = bl; cur_ll[k] = bll;
                        lg = bg; lgl = bgl; ll = bl; lll = bll;
                    }
                    if (r == R - 1 && c_start + 4 == C) {
                        final_g = cur_g[3];
                        final_gl = cur_gl[3];
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int c = c_start + k;
                        if (c < C) {
                            const float sim = row_ptr[k];
                            const float cell = (sim - rm) * rhis + (sim - cm[k]) * chis[k];

                            const float ug = cur_g[k];
                            const unsigned int ugl = cur_gl[k];
                            const float ul = cur_l[k];
                            const unsigned int ull = cur_ll[k];

                            float bg = dg + cell;
                            unsigned int bgl = dgl + 1u;
                            float gd = ug + g_gap;
                            if (gd > bg) { bg = gd; bgl = ugl + 1u; }
                            float gi = lg + g_gap;
                            if (gi > bg) { bg = gi; bgl = lgl + 1u; }

                            float lm = dl + (cell - l_shift);
                            float bl = 0.0f;
                            unsigned int bll = 0u;
                            if (lm > 0.0f) { bl = lm; bll = dll + 1u; }
                            float ld = ul + l_gap;
                            if (ld > bl) { bl = ld; bll = ull + 1u; }
                            float li = ll + l_gap;
                            if (li > bl) { bl = li; bll = lll + 1u; }

                            if (bl > best_l) {
                                best_l = bl; best_ll = bll;
                                best_pos = ((unsigned long long)(unsigned int)r << 32) | (unsigned int)c;
                            } else if (bl == best_l && bl > 0.0f) {
                                unsigned long long pos = ((unsigned long long)(unsigned int)r << 32) | (unsigned int)c;
                                if (pos < best_pos) { best_ll = bll; best_pos = pos; }
                            }

                            dg = ug; dgl = ugl; dl = ul; dll = ull;
                            cur_g[k] = bg; cur_gl[k] = bgl; cur_l[k] = bl; cur_ll[k] = bll;
                            lg = bg; lgl = bgl; ll = bl; lll = bll;

                            if (r == R - 1 && c == C - 1) {
                                final_g = bg;
                                final_gl = bgl;
                            }
                        }
                    }
                }
                send_g = cur_g[3];
                send_gl = cur_gl[3];
                send_l = cur_l[3];
                send_ll = cur_ll[3];
                if (lane == 31 && col_base + swath_w < C) {
                    v_g[r] = send_g;
                    v_gl[r] = send_gl;
                    v_l[r] = send_l;
                    v_ll[r] = send_ll;
                }
            }
        }
    }

#ifdef __HIP_PLATFORM_AMD__
    s_shfl_l[warp_id][lane] = best_l;
    s_shfl_ll[warp_id][lane] = best_ll;
    s_shfl_pos[warp_id][lane] = best_pos;
    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (lane < offset) {
            float ol = s_shfl_l[warp_id][lane + offset];
            unsigned int oll = s_shfl_ll[warp_id][lane + offset];
            unsigned long long opos = s_shfl_pos[warp_id][lane + offset];
            if (ol > s_shfl_l[warp_id][lane] || (ol == s_shfl_l[warp_id][lane] && opos < s_shfl_pos[warp_id][lane])) {
                s_shfl_l[warp_id][lane] = ol;
                s_shfl_ll[warp_id][lane] = oll;
                s_shfl_pos[warp_id][lane] = opos;
            }
        }
        __syncthreads();
    }
    best_l = s_shfl_l[warp_id][0];
    best_ll = s_shfl_ll[warp_id][0];
#else
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ol = __shfl_down_sync(0xFFFFFFFFu, best_l, offset);
        unsigned int oll = __shfl_down_sync(0xFFFFFFFFu, best_ll, offset);
        unsigned long long opos = __shfl_down_sync(0xFFFFFFFFu, best_pos, offset);
        if (ol > best_l || (ol == best_l && opos < best_pos)) {
            best_l = ol; best_ll = oll; best_pos = opos;
        }
    }
#endif

    if (active_pair) {
        const int last_lane = ((C - 1) % swath_w) >> 2;
        if (lane == last_lane) {
            out_gs[p] = final_g;
            out_gl[p] = final_gl;
        }
        if (lane == 0) {
            out_ls[p] = best_l;
            out_ll[p] = best_ll;
        }
    }
}

/* Kernel 3: Float64-gap exact twin for arbitrary gap penalties or scores above
   float32_exact_local_bound. Matches Network_Kernels.global_local_scores_reuse:
   DP tables hold float32 values, gap additions evaluate in float64, local
   shifted score casts to float32, and max_local_score tracks in float64. */
extern "C" __global__ void align_dp_f64gap_k4(
    const float* __restrict__ mat,
    int mat_stride,
    const int* __restrict__ pair_row_off,
    const int* __restrict__ pair_row_len,
    const int* __restrict__ pair_col_off,
    const int* __restrict__ pair_col_len,
    const int* __restrict__ pair_rstat_off,
    const int* __restrict__ pair_cstat_off,
    const float* __restrict__ row_mean,
    const float* __restrict__ row_his,
    const float* __restrict__ col_mean,
    const float* __restrict__ col_his,
    float* __restrict__ scratch_g,
    unsigned int* __restrict__ scratch_gl,
    float* __restrict__ scratch_l,
    unsigned int* __restrict__ scratch_ll,
    int scratch_stride,
    float* __restrict__ out_gs,
    unsigned int* __restrict__ out_gl,
    double* __restrict__ out_ls,
    unsigned int* __restrict__ out_ll,
    int num_pairs,
    double g_gap_d,
    double l_gap_d,
    double l_shift_d)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int p = blockIdx.x * DP_WARPS_PER_BLK + warp_id;

#ifdef __HIP_PLATFORM_AMD__
    __shared__ float s_shfl_g[DP_WARPS_PER_BLK][32];
    __shared__ unsigned int s_shfl_gl[DP_WARPS_PER_BLK][32];
    __shared__ float s_shfl_l[DP_WARPS_PER_BLK][32];
    __shared__ unsigned int s_shfl_ll[DP_WARPS_PER_BLK][32];
    __shared__ double s_red_l[DP_WARPS_PER_BLK][32];
    __shared__ unsigned long long s_shfl_pos[DP_WARPS_PER_BLK][32];
    __shared__ int s_blk_r[DP_WARPS_PER_BLK];
    __shared__ int s_blk_c[DP_WARPS_PER_BLK];
    const bool active_pair = (p < num_pairs);
#else
    if (p >= num_pairs) return;
    const bool active_pair = true;
#endif

    const int r0 = active_pair ? pair_row_off[p] : 0;
    const int R  = active_pair ? pair_row_len[p] : 0;
    const int c0 = active_pair ? pair_col_off[p] : 0;
    const int C  = active_pair ? pair_col_len[p] : 0;
    const float* rm_ptr   = active_pair ? (row_mean + pair_rstat_off[p]) : row_mean;
    const float* rhis_ptr = active_pair ? (row_his  + pair_rstat_off[p]) : row_his;
    const float* cm_ptr   = active_pair ? (col_mean + pair_cstat_off[p]) : col_mean;
    const float* chis_ptr = active_pair ? (col_his  + pair_cstat_off[p]) : col_his;

    float* v_g = scratch_g + (long long)(active_pair ? p : 0) * scratch_stride;
    unsigned int* v_gl = scratch_gl + (long long)(active_pair ? p : 0) * scratch_stride;
    float* v_l = scratch_l + (long long)(active_pair ? p : 0) * scratch_stride;
    unsigned int* v_ll = scratch_ll + (long long)(active_pair ? p : 0) * scratch_stride;

    double best_l = 0.0;
    unsigned int best_ll = 0u;
    unsigned long long best_pos = 0xFFFFFFFFFFFFFFFFull;
    float final_g = 0.0f;
    unsigned int final_gl = 0u;

    const int swath_w = 128;
#ifdef __HIP_PLATFORM_AMD__
    if (lane == 0) {
        s_blk_r[warp_id] = R;
        s_blk_c[warp_id] = C;
    }
    __syncthreads();
    int blk_R = s_blk_r[0];
    int blk_C = s_blk_c[0];
    for (int w = 1; w < DP_WARPS_PER_BLK; ++w) {
        if (s_blk_r[w] > blk_R) blk_R = s_blk_r[w];
        if (s_blk_c[w] > blk_C) blk_C = s_blk_c[w];
    }
    const int loop_C = blk_C;
    const int loop_steps = blk_R + 31;
#else
    const int loop_C = C;
    const int loop_steps = R + 31;
#endif

    for (int col_base = 0; col_base < loop_C; col_base += swath_w) {
        const int c_start = col_base + lane * 4;
        float cm[4], chis[4];
        float cur_g[4], cur_l[4];
        unsigned int cur_gl[4], cur_ll[4];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int c = c_start + k;
            bool valid = (c < C);
            cm[k] = valid ? cm_ptr[c] : 0.0f;
            chis[k] = valid ? chis_ptr[c] : 0.0f;
            cur_g[k] = (float)((double)(c + 1) * g_gap_d);
            cur_gl[k] = (unsigned int)(c + 1);
            cur_l[k] = 0.0f;
            cur_ll[k] = 0u;
        }
        float diag0_g = (float)((double)c_start * g_gap_d);
        unsigned int diag0_gl = (unsigned int)c_start;
        float diag0_l = 0.0f;
        unsigned int diag0_ll = 0u;

        float send_g = cur_g[3];
        unsigned int send_gl = cur_gl[3];
        float send_l = cur_l[3];
        unsigned int send_ll = cur_ll[3];

        for (int t = 0; t < loop_steps; ++t) {
#ifdef __HIP_PLATFORM_AMD__
            s_shfl_g[warp_id][lane] = send_g;
            s_shfl_gl[warp_id][lane] = send_gl;
            s_shfl_l[warp_id][lane] = send_l;
            s_shfl_ll[warp_id][lane] = send_ll;
            __syncthreads();
            float lg = (lane > 0) ? s_shfl_g[warp_id][lane - 1] : 0.0f;
            unsigned int lgl = (lane > 0) ? s_shfl_gl[warp_id][lane - 1] : 0u;
            float ll = (lane > 0) ? s_shfl_l[warp_id][lane - 1] : 0.0f;
            unsigned int lll = (lane > 0) ? s_shfl_ll[warp_id][lane - 1] : 0u;
            __syncthreads();
#else
            float lg = __shfl_up_sync(0xFFFFFFFFu, send_g, 1);
            unsigned int lgl = __shfl_up_sync(0xFFFFFFFFu, send_gl, 1);
            float ll = __shfl_up_sync(0xFFFFFFFFu, send_l, 1);
            unsigned int lll = __shfl_up_sync(0xFFFFFFFFu, send_ll, 1);
#endif
            const int r = t - lane;
            if (r >= 0 && r < R && c_start < C) {
                if (lane == 0) {
                    if (col_base == 0) {
                        lg = (float)((double)(r + 1) * g_gap_d);
                        lgl = (unsigned int)(r + 1);
                        ll = 0.0f;
                        lll = 0u;
                    } else {
                        lg = v_g[r];
                        lgl = v_gl[r];
                        ll = v_l[r];
                        lll = v_ll[r];
                    }
                }
                const float rm = rm_ptr[r];
                const float rhis = rhis_ptr[r];
                const float* row_ptr = mat + (long long)(r0 + r) * mat_stride + (c0 + c_start);

                float dg = diag0_g;
                unsigned int dgl = diag0_gl;
                float dl = diag0_l;
                unsigned int dll = diag0_ll;

                diag0_g = lg; diag0_gl = lgl; diag0_l = ll; diag0_ll = lll;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int c = c_start + k;
                    if (c < C) {
                        const float sim = row_ptr[k];
                        const float cell = (sim - rm) * rhis + (sim - cm[k]) * chis[k];

                        const float ug = cur_g[k];
                        const unsigned int ugl = cur_gl[k];
                        const float ul = cur_l[k];
                        const unsigned int ull = cur_ll[k];

                        double bg_d = (double)dg + (double)cell;
                        unsigned int bgl = dgl + 1u;
                        double gd_d = (double)ug + g_gap_d;
                        if (gd_d > bg_d) { bg_d = gd_d; bgl = ugl + 1u; }
                        double gi_d = (double)lg + g_gap_d;
                        if (gi_d > bg_d) { bg_d = gi_d; bgl = lgl + 1u; }
                        float bg = (float)bg_d;

                        float shifted = (float)((double)cell - l_shift_d);
                        double lm_d = (double)dl + (double)shifted;
                        double bl_d = 0.0;
                        unsigned int bll = 0u;
                        if (lm_d > bl_d) { bl_d = lm_d; bll = dll + 1u; }
                        double ld_d = (double)ul + l_gap_d;
                        if (ld_d > bl_d) { bl_d = ld_d; bll = ull + 1u; }
                        double li_d = (double)ll + l_gap_d;
                        if (li_d > bl_d) { bl_d = li_d; bll = lll + 1u; }
                        float bl = (float)bl_d;

                        if (bl_d > best_l) {
                            best_l = bl_d; best_ll = bll;
                            best_pos = ((unsigned long long)(unsigned int)r << 32) | (unsigned int)c;
                        } else if (bl_d == best_l && bl_d > 0.0) {
                            unsigned long long pos = ((unsigned long long)(unsigned int)r << 32) | (unsigned int)c;
                            if (pos < best_pos) { best_ll = bll; best_pos = pos; }
                        }

                        dg = ug; dgl = ugl; dl = ul; dll = ull;
                        cur_g[k] = bg; cur_gl[k] = bgl; cur_l[k] = bl; cur_ll[k] = bll;
                        lg = bg; lgl = bgl; ll = bl; lll = bll;

                        if (r == R - 1 && c == C - 1) {
                            final_g = bg;
                            final_gl = bgl;
                        }
                    }
                }
                send_g = cur_g[3];
                send_gl = cur_gl[3];
                send_l = cur_l[3];
                send_ll = cur_ll[3];
                if (lane == 31 && col_base + swath_w < C) {
                    v_g[r] = send_g;
                    v_gl[r] = send_gl;
                    v_l[r] = send_l;
                    v_ll[r] = send_ll;
                }
            }
        }
    }

#ifdef __HIP_PLATFORM_AMD__
    s_red_l[warp_id][lane] = best_l;
    s_shfl_ll[warp_id][lane] = best_ll;
    s_shfl_pos[warp_id][lane] = best_pos;
    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
        if (lane < offset) {
            double ol = s_red_l[warp_id][lane + offset];
            unsigned int oll = s_shfl_ll[warp_id][lane + offset];
            unsigned long long opos = s_shfl_pos[warp_id][lane + offset];
            if (ol > s_red_l[warp_id][lane] || (ol == s_red_l[warp_id][lane] && opos < s_shfl_pos[warp_id][lane])) {
                s_red_l[warp_id][lane] = ol;
                s_shfl_ll[warp_id][lane] = oll;
                s_shfl_pos[warp_id][lane] = opos;
            }
        }
        __syncthreads();
    }
    best_l = s_red_l[warp_id][0];
    best_ll = s_shfl_ll[warp_id][0];
#else
    for (int offset = 16; offset > 0; offset >>= 1) {
        double ol = __shfl_down_sync(0xFFFFFFFFu, best_l, offset);
        unsigned int oll = __shfl_down_sync(0xFFFFFFFFu, best_ll, offset);
        unsigned long long opos = __shfl_down_sync(0xFFFFFFFFu, best_pos, offset);
        if (ol > best_l || (ol == best_l && opos < best_pos)) {
            best_l = ol; best_ll = oll; best_pos = opos;
        }
    }
#endif

    if (active_pair) {
        const int last_lane = ((C - 1) % swath_w) >> 2;
        if (lane == last_lane) {
            out_gs[p] = final_g;
            out_gl[p] = final_gl;
        }
        if (lane == 0) {
            out_ls[p] = best_l;
            out_ll[p] = best_ll;
        }
    }
}

/* Kernel 4: Fixed-query SSEARCH DP (local or global) tracking packed
   (length:19 | aligned:15 | similar:15 | identity:15) in uint64, matching
   utilities.Network_Kernels.search_batch_scores.
   mode == 0 -> global, mode == 1 -> local. */
extern "C" __global__ void search_dp_k4(
    const float* __restrict__ mat,
    int mat_stride,
    const int* __restrict__ pair_col_off,
    const int* __restrict__ pair_col_len,
    const int* __restrict__ pair_rstat_off,
    const int* __restrict__ pair_cstat_off,
    const float* __restrict__ row_mean,
    const float* __restrict__ row_his,
    const float* __restrict__ col_mean,
    const float* __restrict__ col_his,
    const unsigned char* __restrict__ query_codes,
    const unsigned char* __restrict__ target_codes,
    const unsigned char* __restrict__ similar_table,
    float* __restrict__ scratch_s,
    unsigned long long* __restrict__ scratch_m,
    int scratch_stride,
    float* __restrict__ out_score,
    unsigned long long* __restrict__ out_meta,
    int R,
    int num_pairs,
    int is_local,
    double gap_d,
    double shift_d)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int p = blockIdx.x * DP_WARPS_PER_BLK + warp_id;

#ifdef __HIP_PLATFORM_AMD__
    __shared__ float s_s[DP_WARPS_PER_BLK][32];
    __shared__ unsigned long long s_m[DP_WARPS_PER_BLK][32];
    __shared__ unsigned long long s_pos[DP_WARPS_PER_BLK][32];
    __shared__ int s_blk_c[DP_WARPS_PER_BLK];
    const bool active_pair = (p < num_pairs);
#else
    if (p >= num_pairs) return;
    const bool active_pair = true;
#endif

    const float gap = (float)gap_d;
    const float shift = (float)shift_d;
    const unsigned long long STEP_LENGTH = 1ull << 45;
    const unsigned long long STEP_ALIGNED = 1ull << 30;
    const unsigned long long STEP_SIMILAR = 1ull << 15;
    const unsigned long long STEP_DIAG = STEP_LENGTH | STEP_ALIGNED;

    const int c0 = active_pair ? pair_col_off[p] : 0;
    const int C  = active_pair ? pair_col_len[p] : 0;
    const float* rm_ptr   = active_pair ? (row_mean + pair_rstat_off[p]) : row_mean;
    const float* rhis_ptr = active_pair ? (row_his  + pair_rstat_off[p]) : row_his;
    const float* cm_ptr   = active_pair ? (col_mean + pair_cstat_off[p]) : col_mean;
    const float* chis_ptr = active_pair ? (col_his  + pair_cstat_off[p]) : col_his;
    const unsigned char* tc_ptr = active_pair ? (target_codes + c0) : target_codes;

    float* v_s = scratch_s + (long long)(active_pair ? p : 0) * scratch_stride;
    unsigned long long* v_m = scratch_m + (long long)(active_pair ? p : 0) * scratch_stride;

    float best_s = 0.0f;
    unsigned long long best_m = 0ull;
    unsigned long long best_pos = 0xFFFFFFFFFFFFFFFFull;
    float final_s = 0.0f;
    unsigned long long final_m = 0ull;

    const int swath_w = 128;
    const int total_steps = R + 31;
#ifdef __HIP_PLATFORM_AMD__
    if (lane == 0) {
        s_blk_c[warp_id] = C;
    }
    __syncthreads();
    int blk_C = s_blk_c[0];
    for (int w = 1; w < DP_WARPS_PER_BLK; ++w) {
        if (s_blk_c[w] > blk_C) blk_C = s_blk_c[w];
    }
    const int loop_C = blk_C;
#else
    const int loop_C = C;
#endif

    for (int col_base = 0; col_base < loop_C; col_base += swath_w) {
        const int c_start = col_base + lane * 4;
        float cm[4], chis[4], cur_s[4];
        unsigned long long cur_m[4];
        unsigned char tc[4];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int c = c_start + k;
            bool valid = (c < C);
            cm[k] = valid ? cm_ptr[c] : 0.0f;
            chis[k] = valid ? chis_ptr[c] : 0.0f;
            tc[k] = valid ? tc_ptr[c] : (unsigned char)1;
            if (is_local) {
                cur_s[k] = 0.0f;
                cur_m[k] = 0ull;
            } else {
                cur_s[k] = (float)(c + 1) * gap;
                cur_m[k] = (unsigned long long)(c + 1) * STEP_LENGTH;
            }
        }
        float diag0_s = is_local ? 0.0f : (float)c_start * gap;
        unsigned long long diag0_m = is_local ? 0ull : (unsigned long long)c_start * STEP_LENGTH;
        float send_s = cur_s[3];
        unsigned long long send_m = cur_m[3];

        for (int t = 0; t < total_steps; ++t) {
#ifdef __HIP_PLATFORM_AMD__
            s_s[warp_id][lane] = send_s;
            s_m[warp_id][lane] = send_m;
            __syncthreads();
            float ls = (lane > 0) ? s_s[warp_id][lane - 1] : 0.0f;
            unsigned long long lm_val = (lane > 0) ? s_m[warp_id][lane - 1] : 0ull;
            __syncthreads();
#else
            float ls = __shfl_up_sync(0xFFFFFFFFu, send_s, 1);
            unsigned long long lm_val = __shfl_up_sync(0xFFFFFFFFu, send_m, 1);
#endif
            const int r = t - lane;
            if (r >= 0 && r < R && c_start < C) {
                if (lane == 0) {
                    if (col_base == 0) {
                        ls = is_local ? 0.0f : (float)(r + 1) * gap;
                        lm_val = is_local ? 0ull : (unsigned long long)(r + 1) * STEP_LENGTH;
                    } else {
                        ls = v_s[r];
                        lm_val = v_m[r];
                    }
                }
                const float rm = rm_ptr[r];
                const float rhis = rhis_ptr[r];
                const unsigned char qc = query_codes[r];
                const unsigned char* sim_row = similar_table + ((int)qc << 8);
                const float* row_ptr = mat + (long long)r * mat_stride + (c0 + c_start);

                float ds = diag0_s;
                unsigned long long dm = diag0_m;
                diag0_s = ls;
                diag0_m = lm_val;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int c = c_start + k;
                    if (c < C) {
                        const float sim = row_ptr[k];
                        const float cell = (sim - rm) * rhis + (sim - cm[k]) * chis[k];
                        const float us = cur_s[k];
                        const unsigned long long um = cur_m[k];

                        unsigned long long step_meta = STEP_DIAG;
                        if (qc == tc[k]) step_meta += 1ull;
                        if (sim_row[tc[k]]) step_meta += STEP_SIMILAR;

                        float bs;
                        unsigned long long bm;
                        if (is_local) {
                            float ms = ds + (cell - shift);
                            bs = 0.0f;
                            bm = 0ull;
                            if (ms > 0.0f) { bs = ms; bm = dm + step_meta; }
                            float del_s = us + gap;
                            if (del_s > bs) { bs = del_s; bm = um + STEP_LENGTH; }
                            float ins_s = ls + gap;
                            if (ins_s > bs) { bs = ins_s; bm = lm_val + STEP_LENGTH; }

                            if (bs > best_s) {
                                best_s = bs; best_m = bm;
                                best_pos = ((unsigned long long)(unsigned int)r << 32) | (unsigned int)c;
                            } else if (bs == best_s && bs > 0.0f) {
                                unsigned long long pos = ((unsigned long long)(unsigned int)r << 32) | (unsigned int)c;
                                if (pos < best_pos) { best_m = bm; best_pos = pos; }
                            }
                        } else {
                            bs = ds + cell;
                            bm = dm + step_meta;
                            float del_s = us + gap;
                            if (del_s > bs) { bs = del_s; bm = um + STEP_LENGTH; }
                            float ins_s = ls + gap;
                            if (ins_s > bs) { bs = ins_s; bm = lm_val + STEP_LENGTH; }
                            if (r == R - 1 && c == C - 1) {
                                final_s = bs;
                                final_m = bm;
                            }
                        }

                        ds = us; dm = um;
                        cur_s[k] = bs; cur_m[k] = bm;
                        ls = bs; lm_val = bm;
                    }
                }
                send_s = cur_s[3];
                send_m = cur_m[3];
                if (lane == 31 && col_base + swath_w < C) {
                    v_s[r] = send_s;
                    v_m[r] = send_m;
                }
            }
        }
    }

    if (is_local) {
#ifdef __HIP_PLATFORM_AMD__
        s_s[warp_id][lane] = best_s;
        s_m[warp_id][lane] = best_m;
        s_pos[warp_id][lane] = best_pos;
        __syncthreads();
        for (int offset = 16; offset > 0; offset >>= 1) {
            if (lane < offset) {
                float os = s_s[warp_id][lane + offset];
                unsigned long long om = s_m[warp_id][lane + offset];
                unsigned long long opos = s_pos[warp_id][lane + offset];
                if (os > s_s[warp_id][lane] || (os == s_s[warp_id][lane] && opos < s_pos[warp_id][lane])) {
                    s_s[warp_id][lane] = os;
                    s_m[warp_id][lane] = om;
                    s_pos[warp_id][lane] = opos;
                }
            }
            __syncthreads();
        }
        best_s = s_s[warp_id][0];
        best_m = s_m[warp_id][0];
#else
        for (int offset = 16; offset > 0; offset >>= 1) {
            float os = __shfl_down_sync(0xFFFFFFFFu, best_s, offset);
            unsigned long long om = __shfl_down_sync(0xFFFFFFFFu, best_m, offset);
            unsigned long long opos = __shfl_down_sync(0xFFFFFFFFu, best_pos, offset);
            if (os > best_s || (os == best_s && opos < best_pos)) {
                best_s = os; best_m = om; best_pos = opos;
            }
        }
#endif
        if (active_pair && lane == 0) {
            out_score[p] = best_s;
            out_meta[p] = best_m;
        }
    } else if (active_pair) {
        const int last_lane = ((C - 1) % swath_w) >> 2;
        if (lane == last_lane) {
            out_score[p] = final_s;
            out_meta[p] = final_m;
        }
    }
}
"""

KERNEL_NAMES = (
    "align_exp_stats",
    "align_dp_f32_k4",
    "align_dp_f64gap_k4",
    "search_dp_k4",
)


_LOCK = threading.RLock()
_KERNELS = {}
_REASONS = {}


def kernels_disabled():
    value = os.environ.get(DISABLE_ENVIRONMENT_VARIABLE, "").strip().lower()
    return value in _OFF_VALUES


def _compile_options(hip, arch):
    if hip:
        return [f"--offload-arch={arch}", "-ffp-contract=off"]
    return [f"--gpu-architecture={arch}", "--fmad=false"]


def _cache_path(compiler, arch):
    folder = _gpu_rt.kernel_cache_folder()
    if not folder:
        return None
    material = "\n".join([
        KERNEL_SOURCE,
        " ".join(_compile_options(compiler.hip, arch)),
        compiler.name,
        compiler.version() or "?",
        str(torch.version.hip or torch.version.cuda),
    ])
    digest = hashlib.sha256(material.encode()).hexdigest()[:24]
    vendor = "hip" if compiler.hip else "cuda"
    safe_arch = re.sub(r"[^A-Za-z0-9_.-]", "_", arch)
    return os.path.join(folder, f"ssn-align-{vendor}-{safe_arch}-{digest}.bin")


class AlignKernels:
    """Compiled and validated alignment kernels for one CUDA/ROCm device."""

    def __init__(self, device, *, read_cache=True):
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        self.device = torch.device("cuda", index)
        self.compiler = _gpu_rt._compiler()
        properties = torch.cuda.get_device_properties(self.device)
        self.device_name = properties.name
        self.multiprocessors = max(1, int(properties.multi_processor_count))
        self.arch = (
            str(properties.gcnArchName)
            if self.compiler.hip
            else f"sm_{properties.major}{properties.minor}"
        )
        self.cache_path = _cache_path(self.compiler, self.arch)
        self.from_cache = False
        self._image = None
        with torch.cuda.device(self.device):
            torch.empty(1, device=self.device)
            if read_cache and self.cache_path:
                self._load_cached()
            if not self.from_cache:
                if self.compiler.hip:
                    self.target = self.arch
                    self._image, _ = self.compiler.compile(
                        KERNEL_SOURCE, _compile_options(True, self.arch)
                    )
                else:
                    self.target, self._image = self._compile_cuda(properties)
                self._module, self._functions = self.compiler.load(
                    self._image, KERNEL_NAMES
                )

    def _compile_cuda(self, properties):
        capability = properties.major * 10 + properties.minor
        try:
            target = f"sm_{capability}"
            image, _ = self.compiler.compile(
                KERNEL_SOURCE, _compile_options(False, target)
            )
            return target, image
        except RuntimeError as native_error:
            known = [
                arch
                for arch in self.compiler.supported_cuda_archs()
                if arch <= capability
            ]
            if not known:
                raise native_error
            target = f"compute_{known[-1]}"
            image, _ = self.compiler.compile(
                KERNEL_SOURCE, _compile_options(False, target)
            )
            return target, image

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

    def launch(self, name, grid, block, args, stream=None):
        if stream is None:
            stream = torch.cuda.current_stream(self.device).cuda_stream
        with torch.cuda.device(self.device):
            self.compiler.launch(self._functions[name], grid, block, args, stream)

    def _score_pairs_direct(
        self,
        mat,
        r_off,
        r_len,
        c_off,
        c_len,
        rs_off,
        cs_off,
        total_r,
        total_c,
        global_gap,
        local_gap,
        *,
        use_f64_gap=False,
    ):
        num_p = int(r_off.shape[0])
        max_r = int(r_len.max().item()) if num_p else 1
        d_rm = torch.empty(max(1, total_r), dtype=torch.float32, device=self.device)
        d_rs = torch.empty(max(1, total_r), dtype=torch.float32, device=self.device)
        d_cm = torch.empty(max(1, total_c), dtype=torch.float32, device=self.device)
        d_cs = torch.empty(max(1, total_c), dtype=torch.float32, device=self.device)
        d_nf = torch.zeros(1, dtype=torch.int32, device=self.device)
        sc_g = torch.empty((num_p, max_r), dtype=torch.float32, device=self.device)
        sc_gl = torch.empty((num_p, max_r), dtype=torch.int32, device=self.device)
        sc_l = torch.empty((num_p, max_r), dtype=torch.float32, device=self.device)
        sc_ll = torch.empty((num_p, max_r), dtype=torch.int32, device=self.device)
        out_gs = torch.empty(num_p, dtype=torch.float32, device=self.device)
        out_gl = torch.empty(num_p, dtype=torch.int32, device=self.device)
        out_ls = torch.empty(
            num_p,
            dtype=torch.float64 if use_f64_gap else torch.float32,
            device=self.device,
        )
        out_ll = torch.empty(num_p, dtype=torch.int32, device=self.device)

        stride = int(mat.stride(0))
        self.launch(
            "align_exp_stats",
            (num_p, 1, 1),
            (128, 1, 1),
            [
                mat, stride, r_off, r_len, c_off, c_len, rs_off, cs_off,
                d_rm, d_rs, d_cm, d_cs, d_nf, num_p,
            ],
        )
        dp_name = "align_dp_f64gap_k4" if use_f64_gap else "align_dp_f32_k4"
        self.launch(
            dp_name,
            ((num_p + 3) // 4, 1, 1),
            (128, 1, 1),
            [
                mat, stride, r_off, r_len, c_off, c_len, rs_off, cs_off,
                d_rm, d_rs, d_cm, d_cs, sc_g, sc_gl, sc_l, sc_ll, max_r,
                out_gs, out_gl, out_ls, out_ll, num_p,
                float(global_gap), float(local_gap), 2.0,
            ],
        )
        torch.cuda.synchronize(self.device)
        return (
            out_gs.cpu().numpy(),
            out_gl.cpu().numpy().astype(np.uint32),
            out_ls.cpu().numpy(),
            out_ll.cpu().numpy().astype(np.uint32),
            int(d_nf.item()),
        )

    def self_check(self):
        """Validate stats, f32 DP, f64-gap DP, search DP, and NaN detection."""
        rng = np.random.default_rng(20261123)
        # Cover lengths <= 4, across 128-col swath boundary (129, 145), singleton (1), and asymmetric pairs.
        lengths = [1, 7, 33, 128, 145, 260]
        dim = 64
        embs = [rng.standard_normal((L, dim)).astype(np.float32) for L in lengths]

        with torch.inference_mode():
            norm_t = [
                torch.nn.functional.normalize(
                    torch.from_numpy(e).to(self.device), p=2, dim=-1
                )
                for e in embs
            ]
            packed = torch.cat(norm_t, dim=0)
            starts = np.cumsum([0] + lengths[:-1]).tolist()
            pairs = [(0, 0)] + [
                (i, j)
                for i in range(len(lengths))
                for j in range(i + 1, len(lengths))
            ]
            num_p = len(pairs)
            r_off = torch.tensor([starts[i] for i, _ in pairs], dtype=torch.int32, device=self.device)
            r_len = torch.tensor([lengths[i] for i, _ in pairs], dtype=torch.int32, device=self.device)
            c_off = torch.tensor([starts[j] for _, j in pairs], dtype=torch.int32, device=self.device)
            c_len = torch.tensor([lengths[j] for _, j in pairs], dtype=torch.int32, device=self.device)
            rs_off = torch.zeros(num_p, dtype=torch.int32, device=self.device)
            rs_off[1:] = torch.cumsum(r_len[:-1], dim=0, dtype=torch.int32)
            cs_off = torch.zeros(num_p, dtype=torch.int32, device=self.device)
            cs_off[1:] = torch.cumsum(c_len[:-1], dim=0, dtype=torch.int32)
            total_r = int(r_len.sum().item())
            total_c = int(c_len.sum().item())

            # Reference CPU score matrices
            ref_mats = []
            for i, j in pairs:
                ti = norm_t[i].cpu().double()
                tj = norm_t[j].cpu().double()
                cos = torch.mm(ti, tj.T).clamp(-1.0, 1.0)
                sim = torch.exp(-(1.0 - cos))
                rm = sim.mean(dim=1, keepdim=True)
                rs = sim.std(dim=1, keepdim=True, correction=0)
                cm = sim.mean(dim=0, keepdim=True)
                cs = sim.std(dim=0, keepdim=True, correction=0)
                sc = (((sim - rm) / (rs + 1e-8)) + ((sim - cm) / (cs + 1e-8))) / 2.0
                ref_mats.append(sc.float().numpy())

            # Test both f32-exact gaps and f64-gap custom penalties
            for g_gap, l_gap, use_f64 in ((0.0, -2.0, False), (-0.7, -1.3, True)):
                mat = torch.mm(packed, packed.T)
                gs, gl, ls, ll, nf = self._score_pairs_direct(
                    mat, r_off, r_len, c_off, c_len, rs_off, cs_off,
                    total_r, total_c, g_gap, l_gap, use_f64_gap=use_f64,
                )
                if nf != 0:
                    raise RuntimeError("self-check flagged finite cosine values as non-finite")
                for idx, ref_m in enumerate(ref_mats):
                    rgs, rgl, rls, rll = global_local_scores(ref_m, g_gap, l_gap)
                    g_scale = max(1, int(rgl))
                    l_scale = max(1, int(rll))
                    if int(gl[idx]) != int(rgl) or int(ll[idx]) != int(rll):
                        raise RuntimeError(
                            f"self-check length mismatch on pair {pairs[idx]} "
                            f"(gaps={g_gap},{l_gap}): got ({gl[idx]},{ll[idx]}) vs ({rgl},{rll})"
                        )
                    if abs(float(gs[idx]) - float(rgs)) / g_scale > _SELF_CHECK_PER_RESIDUE_TOL:
                        raise RuntimeError(f"self-check global score drift on pair {pairs[idx]}")
                    if abs(float(ls[idx]) - float(rls)) / l_scale > _SELF_CHECK_PER_RESIDUE_TOL:
                        raise RuntimeError(f"self-check local score drift on pair {pairs[idx]}")

            # Test NaN detection
            bad_mat = torch.mm(packed, packed.T)
            bad_mat[0, 0] = float("nan")
            _, _, _, _, nf_bad = self._score_pairs_direct(
                bad_mat, r_off[:1], r_len[:1], c_off[:1], c_len[:1],
                rs_off[:1], cs_off[:1], int(r_len[0].item()), int(c_len[0].item()),
                0.0, -2.0, use_f64_gap=False,
            )
            if nf_bad == 0:
                raise RuntimeError("self-check failed to detect NaN in cosine matrix")

            # Test fixed-query search kernel (both local and global)
            q_idx = 4  # length 145
            q_len = lengths[q_idx]
            q_codes = rng.integers(0, 4, size=q_len).astype(np.uint8)
            t_indices = [2, 3, 5]
            t_lens_list = [lengths[t] for t in t_indices]
            t_codes_list = [rng.integers(0, 4, size=L).astype(np.uint8) for L in t_lens_list]
            sim_tbl = np.zeros((256, 256), dtype=np.uint8)
            sim_tbl[:4, :4] = rng.integers(0, 2, size=(4, 4), dtype=np.uint8)

            q_t = norm_t[q_idx]
            t_cat = torch.cat([norm_t[t] for t in t_indices], dim=0)
            s_c_off = np.cumsum([0] + t_lens_list[:-1]).astype(np.int32)
            s_c_len = np.array(t_lens_list, dtype=np.int32)
            s_r_off = np.zeros(len(t_indices), dtype=np.int32)
            s_r_len = np.full(len(t_indices), q_len, dtype=np.int32)
            s_rs_off = (np.arange(len(t_indices), dtype=np.int32) * q_len)
            s_cs_off = s_c_off.copy()

            d_r_off = torch.from_numpy(s_r_off).to(self.device)
            d_r_len = torch.from_numpy(s_r_len).to(self.device)
            d_c_off = torch.from_numpy(s_c_off).to(self.device)
            d_c_len = torch.from_numpy(s_c_len).to(self.device)
            d_rs_off = torch.from_numpy(s_rs_off).to(self.device)
            d_cs_off = torch.from_numpy(s_cs_off).to(self.device)
            d_qc = torch.from_numpy(q_codes).to(self.device)
            d_tc = torch.from_numpy(np.concatenate(t_codes_list)).to(self.device)
            d_st = torch.from_numpy(sim_tbl.reshape(-1)).to(self.device)

            for is_local, gap in ((1, -2.0), (0, 0.0)):
                smat = torch.mm(q_t, t_cat.T)
                tot_r = int(s_r_len.sum())
                tot_c = int(s_c_len.sum())
                d_rm = torch.empty(tot_r, dtype=torch.float32, device=self.device)
                d_rs = torch.empty(tot_r, dtype=torch.float32, device=self.device)
                d_cm = torch.empty(tot_c, dtype=torch.float32, device=self.device)
                d_cs = torch.empty(tot_c, dtype=torch.float32, device=self.device)
                d_nf = torch.zeros(1, dtype=torch.int32, device=self.device)
                sc_s = torch.empty((len(t_indices), q_len), dtype=torch.float32, device=self.device)
                sc_m = torch.empty((len(t_indices), q_len), dtype=torch.int64, device=self.device)
                o_s = torch.empty(len(t_indices), dtype=torch.float32, device=self.device)
                o_m = torch.empty(len(t_indices), dtype=torch.int64, device=self.device)
                self.launch(
                    "align_exp_stats", (len(t_indices), 1, 1), (128, 1, 1),
                    [smat, int(smat.stride(0)), d_r_off, d_r_len, d_c_off, d_c_len,
                     d_rs_off, d_cs_off, d_rm, d_rs, d_cm, d_cs, d_nf, len(t_indices)],
                )
                self.launch(
                    "search_dp_k4", ((len(t_indices) + 3) // 4, 1, 1), (128, 1, 1),
                    [smat, int(smat.stride(0)), d_c_off, d_c_len, d_rs_off, d_cs_off,
                     d_rm, d_rs, d_cm, d_cs, d_qc, d_tc, d_st, sc_s, sc_m, q_len,
                     o_s, o_m, q_len, len(t_indices), int(is_local), float(gap), 2.0],
                )
                torch.cuda.synchronize(self.device)
                got_s = o_s.cpu().numpy()
                got_m = o_m.cpu().numpy().view(np.uint64)
                for k_i, t_idx in enumerate(t_indices):
                    ti = q_t.cpu().double()
                    tj = norm_t[t_idx].cpu().double()
                    cos = torch.mm(ti, tj.T).clamp(-1.0, 1.0)
                    sim = torch.exp(-(1.0 - cos))
                    rm = sim.mean(dim=1, keepdim=True)
                    rs = sim.std(dim=1, keepdim=True, correction=0)
                    cm = sim.mean(dim=0, keepdim=True)
                    cs = sim.std(dim=0, keepdim=True, correction=0)
                    ref_m = ((((sim - rm) / (rs + 1e-8)) + ((sim - cm) / (cs + 1e-8))) / 2.0).float().numpy()
                    fn = local_score_length_identity if is_local else global_score_length_identity
                    rs_val, rlen, rid, rsim, raln = fn(ref_m, gap, q_codes, t_codes_list[k_i], sim_tbl)
                    packed_m = int(got_m[k_i])
                    glen = packed_m >> 45
                    galn = (packed_m >> 30) & 0x7FFF
                    gsim = (packed_m >> 15) & 0x7FFF
                    gid = packed_m & 0x7FFF
                    if (glen, gid, gsim, galn) != (int(rlen), int(rid), int(rsim), int(raln)):
                        raise RuntimeError(
                            f"search self-check meta mismatch (local={is_local}): "
                            f"got {(glen, gid, gsim, galn)} vs {(int(rlen), int(rid), int(rsim), int(raln))}"
                        )
                    if abs(float(got_s[k_i]) - float(rs_val)) / max(1, int(rlen)) > _SELF_CHECK_PER_RESIDUE_TOL:
                        raise RuntimeError(f"search self-check score drift (local={is_local})")


def align_kernels(device):
    """Return validated AlignKernels for a CUDA/ROCm device, or None."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    if kernels_disabled():
        return None
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    with _LOCK:
        if index not in _KERNELS:
            try:
                kernels = AlignKernels(device)
                try:
                    kernels.self_check()
                except Exception:
                    if not kernels.from_cache:
                        raise
                    kernels.discard_cache()
                    kernels = AlignKernels(device, read_cache=False)
                    kernels.self_check()
                kernels.store_in_cache()
                _KERNELS[index] = kernels
            except Exception as error:
                _KERNELS[index] = None
                _REASONS[index] = f"{type(error).__name__}: {error}"
        return _KERNELS[index]


def fallback_reason(device):
    device = torch.device(device)
    if device.type != "cuda":
        return f"no runtime alignment kernels for {device.type} devices"
    if kernels_disabled():
        return f"disabled by {DISABLE_ENVIRONMENT_VARIABLE}"
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    return _REASONS.get(index)


def _choose_block_residues(usable_bytes, n_streams, max_seq_len, matrix_budget_override=None):
    """Choose a VRAM-safe GEMM residue span M (matrix = M*M*4 bytes per stream)."""
    # Keep total matrix buffers <= ~25% of usable VRAM, capped at DEFAULT_BLOCK_RESIDUES.
    per_stream_budget = max(16 * 1024 * 1024, int(usable_bytes * 0.25) // max(1, int(n_streams)))
    if matrix_budget_override is not None and int(matrix_budget_override) > 0:
        per_stream_budget = min(per_stream_budget, max(4 * 1024 * 1024, int(matrix_budget_override)))
    max_side = int(math.sqrt(per_stream_budget // 4))
    # Round down to a multiple of 512 for clean GEMM shapes
    side = max(MIN_BLOCK_RESIDUES, min(DEFAULT_BLOCK_RESIDUES, (max_side // 512) * 512))
    return max(side, int(max_seq_len))


def _pack_residue_blocks(indices, lengths, block_res):
    """Partition sorted sequence indices into contiguous blocks of <= block_res rows."""
    blocks = []
    cur = []
    cur_rows = 0
    for idx in indices:
        L = int(lengths[idx])
        if cur and cur_rows + L > block_res:
            blocks.append(cur)
            cur = []
            cur_rows = 0
        cur.append(int(idx))
        cur_rows += L
    if cur:
        blocks.append(cur)
    return blocks


_UPLOAD_CHUNK_BYTES = 32 * 1024 * 1024  # Bound host & GPU staging to <= 32 MiB per chunk


def _upload_and_normalize_indices(indices, lengths, feature_dim, store, group, device, compute_dtype):
    """Upload embeddings for `indices` into one contiguous L2-normalized GPU buffer.

    Streams in small pinned chunks directly into the pre-allocated destination tensor
    so FP16/BF16 -> FP32 conversion never holds both a full raw copy and a full FP32
    copy on GPU (or a multi-hundred-MiB pinned buffer on host) simultaneously.
    """
    starts = {}
    total_rows = 0
    for idx in indices:
        starts[idx] = total_rows
        total_rows += int(lengths[idx])
    if total_rows == 0:
        return torch.empty((0, feature_dim), dtype=compute_dtype, device=device), starts

    gpu_out = torch.empty((total_rows, feature_dim), dtype=compute_dtype, device=device)
    eps = torch.finfo(torch.float32).tiny
    common_dtype = store.dtypes[indices[0]] if hasattr(store, "dtypes") and store.dtypes else None
    same_dtype = (
        common_dtype is not None
        and all(store.dtypes[idx] == common_dtype for idx in indices)
    )
    torch_dtypes = {
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
    }

    def _normalize_chunk_into(host_slice, dst_start, n_rows, non_blocking):
        dst_slice = gpu_out[dst_start:dst_start + n_rows]
        if compute_dtype == torch.float32:
            if host_slice.dtype == torch.float32:
                dst_slice.copy_(host_slice, non_blocking=non_blocking)
            else:
                dev_raw = host_slice.to(device=device, non_blocking=non_blocking)
                dst_slice.copy_(dev_raw)
                del dev_raw
            norms = torch.linalg.vector_norm(dst_slice, ord=2, dim=-1, keepdim=True)
            dst_slice.div_(norms.clamp_min_(eps))
            del norms
        else:
            dev_raw = host_slice.to(device=device, non_blocking=non_blocking)
            dev_f32 = dev_raw if dev_raw.dtype == torch.float32 else dev_raw.float()
            if dev_raw is not dev_f32:
                del dev_raw
            norms = torch.linalg.vector_norm(dev_f32, ord=2, dim=-1, keepdim=True)
            dev_f32.div_(norms.clamp_min_(eps))
            dst_slice.copy_(dev_f32)
            del dev_f32, norms

    if same_dtype and np.dtype(common_dtype) in torch_dtypes:
        t_dtype = torch_dtypes[np.dtype(common_dtype)]
        src_itemsize = max(1, np.dtype(common_dtype).itemsize)
        # Keep each staging buffer <= 16 MiB (2 buffers = <= 32 MiB host RAM total).
        per_buf_bytes = max(1, _UPLOAD_CHUNK_BYTES // 2)
        chunk_rows = max(1, min(total_rows, per_buf_bytes // max(1, feature_dim * max(src_itemsize, 4))))
        non_blocking = False
        try:
            host_bufs = [
                torch.empty((chunk_rows, feature_dim), dtype=t_dtype, pin_memory=True)
                for _ in range(2)
            ]
            non_blocking = True
        except RuntimeError:
            host_bufs = [
                torch.empty((chunk_rows, feature_dim), dtype=t_dtype)
                for _ in range(2)
            ]
        host_nps = [b.numpy() for b in host_bufs]
        buf_events = [None, None]
        cur_buf = 0
        fill = 0
        dst_cursor = 0

        def _flush_buf():
            nonlocal cur_buf, fill, dst_cursor
            if fill == 0:
                return
            _normalize_chunk_into(host_bufs[cur_buf][:fill], dst_cursor, fill, non_blocking)
            if non_blocking:
                ev = torch.cuda.Event()
                ev.record()
                buf_events[cur_buf] = ev
            dst_cursor += fill
            fill = 0
            cur_buf ^= 1
            if buf_events[cur_buf] is not None:
                buf_events[cur_buf].synchronize()
                buf_events[cur_buf] = None

        for idx in indices:
            arr = store.get(idx, group)
            L = int(lengths[idx])
            pos = 0
            while pos < L:
                space = chunk_rows - fill
                take = min(space, L - pos)
                host_nps[cur_buf][fill:fill + take] = arr[pos:pos + take]
                fill += take
                pos += take
                if fill >= chunk_rows:
                    _flush_buf()
        _flush_buf()
        for ev in buf_events:
            if ev is not None:
                ev.synchronize()
    else:
        host_embs = store.load_indices(indices, group)
        chunk_rows = max(1, min(total_rows, _UPLOAD_CHUNK_BYTES // max(1, feature_dim * 4)))
        dst_cursor = 0
        for idx in indices:
            arr = np.ascontiguousarray(host_embs[idx], dtype=np.float32)
            L = int(arr.shape[0])
            pos = 0
            while pos < L:
                take = min(chunk_rows, L - pos)
                _normalize_chunk_into(torch.from_numpy(arr[pos:pos + take]), dst_cursor, take, False)
                dst_cursor += take
                pos += take
    return gpu_out, starts


def run_fused_gpu_alignment_pipeline(
    tasks,
    *,
    store,
    lengths,
    device,
    lanes,
    global_gap,
    local_gap,
    precision="float32",
    progress=None,
    result_callback=None,
    result_chunk_size=65536,
    matrix_budget_override=None,
    memory_plan_override=None,
    benchmark_trial=None,
    cuda_matmul_precision_ctx=None,
    precision_compute_dtype_fn=None,
    precision_element_bytes_fn=None,
    accelerator_memory_plan_fn=None,
    allocator_cap_ctx=None,
    clear_failed_frames_fn=None,
    partition_tiles_fn=None,
    backend=None,
):
    """Execute pairwise alignment end-to-end on GPU with zero PCIe score-matrix transfer."""
    kernels = align_kernels(device)
    if kernels is None:
        raise RuntimeError(f"Fused GPU alignment kernels unavailable on '{device}'.")
    tasks = list(tasks)
    if not tasks:
        return []

    element_bytes = precision_element_bytes_fn(precision)
    compute_dtype = precision_compute_dtype_fn(precision)
    plan = memory_plan_override or accelerator_memory_plan_fn(
        device, lanes=lanes, compute_element_bytes=element_bytes
    )
    n_streams = max(1, min(2, int(lanes)))
    feature_dim = int(store.feature_dimension)
    tile_cache_budget = int(plan.tile_cache_bytes)
    is_oom_retry = (
        matrix_budget_override is not None
        and int(matrix_budget_override) > 0
        and int(plan.matrix_bytes) > 0
        and int(matrix_budget_override) < int(plan.matrix_bytes)
    )
    if is_oom_retry:
        ratio = float(matrix_budget_override) / float(plan.matrix_bytes)
        max_task_len = max(max(int(lengths[int(t[0])]), int(lengths[int(t[1])])) for t in tasks)
        min_tile_bytes = max(1, max_task_len * 2 * feature_dim * element_bytes)
        tile_cache_budget = max(min_tile_bytes, min(tile_cache_budget, int(tile_cache_budget * ratio)))
    per_block = max(1, tile_cache_budget // 2)
    block_ids = store.block_ids(per_block, element_bytes=element_bytes)
    exact_bound = float32_exact_local_bound(global_gap, local_gap)
    use_f64_always = exact_bound is None
    bound_val = float(exact_bound) if exact_bound is not None else 0.0

    results = []
    active_trial = None

    def emit_results(batch_rows):
        if not batch_rows:
            return
        if progress is not None:
            progress.update(len(batch_rows))
        if active_trial is not None:
            active_trial.completed += len(batch_rows)
        if result_callback is not None:
            results.extend(batch_rows)
            if len(results) >= int(result_chunk_size):
                result_callback(results)
                results.clear()
        else:
            results.extend(batch_rows)

    streams = [torch.cuda.Stream(device=device) for _ in range(n_streams)]
    d_nf = torch.zeros(1, dtype=torch.int32, device=device)
    ws_list = [{} for _ in range(n_streams)]
    cap_stack = ExitStack()
    if allocator_cap_ctx is not None and backend is not None:
        cap_stack.enter_context(allocator_cap_ctx(backend, plan))

    h5_ctx = (
        nullcontext(None)
        if getattr(store, "fully_cached", False)
        else h5py.File(store.path, "r", libver="latest", swmr=True)
    )

    try:
        with cuda_matmul_precision_ctx(precision), torch.inference_mode(), h5_ctx as hf:
            group = None if hf is None else hf["embeddings"]

            def ensure_ws(ws, m_dim, n_dim, num_p, tot_r, tot_c, max_r, max_mat_elems):
                need_mat = max(m_dim * n_dim, max_mat_elems)
                if "mat" not in ws or ws["mat"].numel() < m_dim * n_dim:
                    ws["mat"] = torch.empty(need_mat, dtype=torch.float32, device=device)
                if "rm" not in ws or ws["rm"].shape[0] < tot_r:
                    sz = max(tot_r, 16384)
                    ws["rm"] = torch.empty(sz, dtype=torch.float32, device=device)
                    ws["rs"] = torch.empty(sz, dtype=torch.float32, device=device)
                if "cm" not in ws or ws["cm"].shape[0] < tot_c:
                    sz = max(tot_c, 16384)
                    ws["cm"] = torch.empty(sz, dtype=torch.float32, device=device)
                    ws["cs"] = torch.empty(sz, dtype=torch.float32, device=device)
                sc_need = num_p * max_r
                if "sc_g" not in ws or ws["sc_g"].numel() < sc_need:
                    sz = max(sc_need, 262144)
                    ws["sc_g"] = torch.empty(sz, dtype=torch.float32, device=device)
                    ws["sc_gl"] = torch.empty(sz, dtype=torch.int32, device=device)
                    ws["sc_l"] = torch.empty(sz, dtype=torch.float32, device=device)
                    ws["sc_ll"] = torch.empty(sz, dtype=torch.int32, device=device)
                if "ogs" not in ws or ws["ogs"].shape[0] < num_p:
                    sz = max(num_p, 4096)
                    ws["ogs"] = torch.empty(sz, dtype=torch.float32, device=device)
                    ws["ogl"] = torch.empty(sz, dtype=torch.int32, device=device)
                    ws["ols"] = torch.empty(
                        sz, dtype=torch.float64 if use_f64_always else torch.float32, device=device
                    )
                    ws["oll"] = torch.empty(sz, dtype=torch.int32, device=device)
                    ws["h_gs"] = torch.empty(sz, dtype=torch.float32, pin_memory=True)
                    ws["h_gl"] = torch.empty(sz, dtype=torch.int32, pin_memory=True)
                    ws["h_ls"] = torch.empty(
                        sz, dtype=torch.float64 if use_f64_always else torch.float32, pin_memory=True
                    )
                    ws["h_ll"] = torch.empty(sz, dtype=torch.int32, pin_memory=True)
                    ws["h_meta"] = torch.empty((6, sz), dtype=torch.int32, pin_memory=True)
                    ws["d_meta"] = torch.empty((6, sz), dtype=torch.int32, device=device)

            def run_phase(phase_tasks):
                phase_tiles = [list(tile) for tile in partition_tiles_fn(phase_tasks, block_ids)]

                for tile_tasks in phase_tiles:
                    if active_trial is not None:
                        if not active_trial.can_submit():
                            break
                        active_trial.tiles += 1

                    # Compute per-sequence degree within this tile so bipartite / hub-and-spoke
                    # workloads (e.g. Network Injection: M new seqs vs N old seqs) keep high-degree
                    # hub sequences packed together in 100%-dense blocks, while sorting each tier
                    # by ascending length to eliminate DP warp tail-latency and minimize swaths.
                    deg = {}
                    for t in tile_tasks:
                        u, v = int(t[0]), int(t[1])
                        deg[u] = deg.get(u, 0) + 1
                        deg[v] = deg.get(v, 0) + 1
                    hub_thresh = 0.5 * max(1, len(deg) - 1)
                    tile_indices = sorted(
                        deg.keys(),
                        key=lambda i: (0 if deg[i] >= hub_thresh else 1, int(lengths[i]), i),
                    )
                    max_tile_len = max(int(lengths[i]) for i in tile_indices)
                    block_res = _choose_block_residues(
                        plan.usable_bytes, n_streams, max_tile_len, matrix_budget_override
                    )
                    max_mat_elems = block_res * block_res

                    # Upload tile embeddings once into one contiguous GPU tensor
                    packed, starts = _upload_and_normalize_indices(
                        tile_indices, lengths, feature_dim, store, group, device, compute_dtype
                    )
                    upload_ev = torch.cuda.Event()
                    upload_ev.record()
                    for st in streams:
                        st.wait_event(upload_ev)
                    # Partition tile_indices into residue blocks of size <= block_res
                    rblocks = _pack_residue_blocks(tile_indices, lengths, block_res)
                    seq_to_blk = {}
                    for b_id, b_seqs in enumerate(rblocks):
                        for s_idx in b_seqs:
                            seq_to_blk[s_idx] = b_id

                    # Build slice metadata and max sequence length per residue block
                    blk_slices = []
                    blk_maxlen = []
                    for b_seqs in rblocks:
                        b0 = starts[b_seqs[0]]
                        b1 = starts[b_seqs[-1]] + int(lengths[b_seqs[-1]])
                        blk_slices.append((packed[b0:b1], b0, b1 - b0))
                        blk_maxlen.append(max(int(lengths[s]) for s in b_seqs))
                    max_blk_rows = max((s[2] for s in blk_slices), default=1)
                    tile_max_mat_elems = min(max_mat_elems, max_blk_rows * max_blk_rows)

                    # Group tasks by canonical (row_blk, col_blk) where row_blk has >= max sequence
                    # length so:
                    # 1. Each pair of blocks computes at most ONE dense GEMM
                    # 2. Putting longer sequences on rows minimizes 128-col swath count & scratch.
                    blk_pairs = OrderedDict()
                    for t in tile_tasks:
                        si, sj = int(t[0]), int(t[1])
                        bi, bj = seq_to_blk[si], seq_to_blk[sj]
                        if (blk_maxlen[bi], bi) >= (blk_maxlen[bj], bj):
                            blk_pairs.setdefault((bi, bj), []).append((si, sj, si, sj))
                        else:
                            blk_pairs.setdefault((bj, bi), []).append((si, sj, sj, si))

                    inflight = []
                    stream_cursor = 0

                    def drain_one():
                        ev, sid, pair_ids, num_p = inflight.pop(0)
                        ev.synchronize()
                        ws = ws_list[sid]
                        gs = ws["h_gs"][:num_p].numpy()
                        gl = ws["h_gl"][:num_p].numpy().view(np.uint32)
                        ls = ws["h_ls"][:num_p].numpy()
                        ll = ws["h_ll"][:num_p].numpy().view(np.uint32)
                        # Check rare f32 bound overflow -> exact f64 recompute if ever triggered
                        if not use_f64_always and bool(np.any(ls >= bound_val)):
                            ls = ls.astype(np.float64, copy=True)
                            for k_idx in np.flatnonzero(ls >= bound_val):
                                si, sj = pair_ids[int(k_idx)]
                                ti = packed[starts[si]:starts[si] + int(lengths[si])].float()
                                tj = packed[starts[sj]:starts[sj] + int(lengths[sj])].float()
                                cos = torch.mm(ti, tj.T).clamp_(-1.0, 1.0)
                                sim = torch.exp(-(1.0 - cos))
                                rm = sim.mean(dim=1, keepdim=True)
                                rs = sim.std(dim=1, keepdim=True, correction=0)
                                cm = sim.mean(dim=0, keepdim=True)
                                cs = sim.std(dim=0, keepdim=True, correction=0)
                                sc = (((sim - rm) / (rs + 1e-8)) + ((sim - cm) / (cs + 1e-8))) / 2.0
                                rgs, rgl, rls, rll = global_local_scores(
                                    sc.cpu().numpy(), global_gap, local_gap
                                )
                                gs[k_idx], gl[k_idx], ls[k_idx], ll[k_idx] = rgs, rgl, rls, rll
                        rows = [
                            (si, sj, l_s, l_l, g_s, g_l)
                            for (si, sj), l_s, l_l, g_s, g_l in zip(
                                pair_ids, ls.tolist(), ll.tolist(), gs.tolist(), gl.tolist()
                            )
                        ]
                        emit_results(rows)

                    def submit_gpu_job(A_ten, B_ten, pair_ids, r_off, r_len, c_off, c_len):
                        nonlocal stream_cursor
                        num_p = len(pair_ids)
                        if active_trial is not None:
                            active_trial.submitted += num_p
                            active_trial.microbatches += 1
                        sid = stream_cursor % n_streams
                        stream_cursor += 1
                        while len(inflight) >= n_streams:
                            drain_one()
                        st = streams[sid]
                        ws = ws_list[sid]
                        M = int(A_ten.shape[0])
                        N = int(B_ten.shape[0])
                        rs_off = np.zeros(num_p, dtype=np.int32)
                        if num_p > 1:
                            rs_off[1:] = np.cumsum(r_len[:-1], dtype=np.int32)
                        cs_off = np.zeros(num_p, dtype=np.int32)
                        if num_p > 1:
                            cs_off[1:] = np.cumsum(c_len[:-1], dtype=np.int32)
                        tot_r = int(r_len.sum())
                        tot_c = int(c_len.sum())
                        max_r = int(r_len.max())

                        with torch.cuda.stream(st):
                            ensure_ws(ws, M, N, num_p, tot_r, tot_c, max_r, tile_max_mat_elems)
                            h_meta = ws["h_meta"][:, :num_p]
                            h_meta_np = h_meta.numpy()
                            h_meta_np[0] = r_off
                            h_meta_np[1] = r_len
                            h_meta_np[2] = c_off
                            h_meta_np[3] = c_len
                            h_meta_np[4] = rs_off
                            h_meta_np[5] = cs_off
                            meta_d = ws["d_meta"][:, :num_p]
                            meta_d.copy_(h_meta, non_blocking=True)
                            mat = ws["mat"][:M * N].view(M, N)
                            if compute_dtype == torch.float32:
                                torch.mm(A_ten, B_ten.T, out=mat)
                            else:
                                mat.copy_(torch.mm(A_ten, B_ten.T).to(torch.float32))
                            kernels.launch(
                                "align_exp_stats",
                                (num_p, 1, 1),
                                (128, 1, 1),
                                [
                                    mat, N, meta_d[0], meta_d[1], meta_d[2], meta_d[3],
                                    meta_d[4], meta_d[5], ws["rm"], ws["rs"], ws["cm"], ws["cs"],
                                    d_nf, num_p,
                                ],
                                st.cuda_stream,
                            )
                            dp_name = "align_dp_f64gap_k4" if use_f64_always else "align_dp_f32_k4"
                            kernels.launch(
                                dp_name,
                                ((num_p + 3) // 4, 1, 1),
                                (128, 1, 1),
                                [
                                    mat, N, meta_d[0], meta_d[1], meta_d[2], meta_d[3],
                                    meta_d[4], meta_d[5], ws["rm"], ws["rs"], ws["cm"], ws["cs"],
                                    ws["sc_g"], ws["sc_gl"], ws["sc_l"], ws["sc_ll"], max_r,
                                    ws["ogs"], ws["ogl"], ws["ols"], ws["oll"],
                                    num_p, float(global_gap), float(local_gap), 2.0,
                                ],
                                st.cuda_stream,
                            )
                            ws["h_gs"][:num_p].copy_(ws["ogs"][:num_p], non_blocking=True)
                            ws["h_gl"][:num_p].copy_(ws["ogl"][:num_p], non_blocking=True)
                            ws["h_ls"][:num_p].copy_(ws["ols"][:num_p], non_blocking=True)
                            ws["h_ll"][:num_p].copy_(ws["oll"][:num_p], non_blocking=True)
                            ev = torch.cuda.Event()
                            ev.record(st)
                        inflight.append((ev, sid, pair_ids, num_p))

                    def submit_sparse_slab_job(slab_chunks, slab_M, slab_N, num_p, slab_mat_cap, slab_b_cap):
                        """Execute many sparse rows in one packed 2D matrix slab on a single stream.

                        Packing multiple rows onto 2D shelves inside one reusable `ws["mat"]` buffer
                        eliminates per-row kernel/sync overhead on sparse prefiltered or injection
                        graphs while keeping target assembly in one reusable per-stream `ws["b_buf"]`
                        (zero dynamic `torch.cat` allocator fragmentation and zero cross-stream races).
                        """
                        nonlocal stream_cursor
                        if active_trial is not None:
                            active_trial.submitted += num_p
                            active_trial.microbatches += 1
                        sid = stream_cursor % n_streams
                        stream_cursor += 1
                        while len(inflight) >= n_streams:
                            drain_one()
                        st = streams[sid]
                        ws = ws_list[sid]

                        pair_ids = []
                        r_off = np.empty(num_p, dtype=np.int32)
                        r_len = np.empty(num_p, dtype=np.int32)
                        c_off = np.empty(num_p, dtype=np.int32)
                        c_len = np.empty(num_p, dtype=np.int32)
                        pos = 0
                        for y0, x0, row_L, _c_cols, _ri, c_targets in slab_chunks:
                            cx = x0
                            for si, sj, cj in c_targets:
                                L_j = int(lengths[cj])
                                pair_ids.append((si, sj))
                                r_off[pos] = y0
                                r_len[pos] = row_L
                                c_off[pos] = cx
                                c_len[pos] = L_j
                                cx += L_j
                                pos += 1

                        rs_off = np.zeros(num_p, dtype=np.int32)
                        if num_p > 1:
                            rs_off[1:] = np.cumsum(r_len[:-1], dtype=np.int32)
                        cs_off = np.zeros(num_p, dtype=np.int32)
                        if num_p > 1:
                            cs_off[1:] = np.cumsum(c_len[:-1], dtype=np.int32)
                        tot_r = int(r_len.sum())
                        tot_c = int(c_len.sum())
                        max_r = int(r_len.max())
                        need_b = max(
                            (c_cols for _, _, _, c_cols, _, ct in slab_chunks if len(ct) > 1),
                            default=0,
                        )
                        mat_hint = max(slab_M * slab_N, slab_mat_cap)

                        with torch.cuda.stream(st):
                            ensure_ws(ws, slab_M, slab_N, num_p, tot_r, tot_c, max_r, mat_hint)
                            if need_b > 0 and (
                                "b_buf" not in ws
                                or ws["b_buf"].shape[0] < need_b
                                or ws["b_buf"].dtype != compute_dtype
                            ):
                                ws["b_buf"] = torch.empty(
                                    (max(need_b, slab_b_cap), feature_dim),
                                    dtype=compute_dtype,
                                    device=device,
                                )
                            h_meta = ws["h_meta"][:, :num_p]
                            h_meta_np = h_meta.numpy()
                            h_meta_np[0] = r_off
                            h_meta_np[1] = r_len
                            h_meta_np[2] = c_off
                            h_meta_np[3] = c_len
                            h_meta_np[4] = rs_off
                            h_meta_np[5] = cs_off
                            meta_d = ws["d_meta"][:, :num_p]
                            meta_d.copy_(h_meta, non_blocking=True)
                            mat = ws["mat"][:slab_M * slab_N].view(slab_M, slab_N)
                            for y0, x0, row_L, c_cols, ri, c_targets in slab_chunks:
                                a_row = packed[starts[ri]:starts[ri] + row_L]
                                if len(c_targets) == 1:
                                    cj = c_targets[0][2]
                                    b_view = packed[starts[cj]:starts[cj] + c_cols]
                                else:
                                    b_view = ws["b_buf"][:c_cols]
                                    torch.cat(
                                        [
                                            packed[starts[x[2]]:starts[x[2]] + int(lengths[x[2]])]
                                            for x in c_targets
                                        ],
                                        dim=0,
                                        out=b_view,
                                    )
                                sub = mat[y0:y0 + row_L, x0:x0 + c_cols]
                                if compute_dtype == torch.float32:
                                    torch.mm(a_row, b_view.T, out=sub)
                                else:
                                    sub.copy_(torch.mm(a_row, b_view.T).to(torch.float32))
                            kernels.launch(
                                "align_exp_stats",
                                (num_p, 1, 1),
                                (128, 1, 1),
                                [
                                    mat, slab_N, meta_d[0], meta_d[1], meta_d[2], meta_d[3],
                                    meta_d[4], meta_d[5], ws["rm"], ws["rs"], ws["cm"], ws["cs"],
                                    d_nf, num_p,
                                ],
                                st.cuda_stream,
                            )
                            dp_name = "align_dp_f64gap_k4" if use_f64_always else "align_dp_f32_k4"
                            kernels.launch(
                                dp_name,
                                ((num_p + 3) // 4, 1, 1),
                                (128, 1, 1),
                                [
                                    mat, slab_N, meta_d[0], meta_d[1], meta_d[2], meta_d[3],
                                    meta_d[4], meta_d[5], ws["rm"], ws["rs"], ws["cm"], ws["cs"],
                                    ws["sc_g"], ws["sc_gl"], ws["sc_l"], ws["sc_ll"], max_r,
                                    ws["ogs"], ws["ogl"], ws["ols"], ws["oll"],
                                    num_p, float(global_gap), float(local_gap), 2.0,
                                ],
                                st.cuda_stream,
                            )
                            ws["h_gs"][:num_p].copy_(ws["ogs"][:num_p], non_blocking=True)
                            ws["h_gl"][:num_p].copy_(ws["ogl"][:num_p], non_blocking=True)
                            ws["h_ls"][:num_p].copy_(ws["ols"][:num_p], non_blocking=True)
                            ws["h_ll"][:num_p].copy_(ws["oll"][:num_p], non_blocking=True)
                            ev = torch.cuda.Event()
                            ev.record(st)
                        inflight.append((ev, sid, pair_ids, num_p))

                    A_ten = B_ten = None
                    try:
                        sparse_pairs = []
                        for (bi, bj), pair_quads in blk_pairs.items():
                            if active_trial is not None and not active_trial.can_submit():
                                break
                            A_ten, a_base, M = blk_slices[bi]
                            B_ten, b_base, N = blk_slices[bj]
                            r_len = np.array([lengths[ri] for _, _, ri, _ in pair_quads], dtype=np.int32)
                            c_len = np.array([lengths[cj] for _, _, _, cj in pair_quads], dtype=np.int32)
                            active_cells = int(np.sum(r_len.astype(np.int64) * c_len.astype(np.int64)))
                            density = active_cells / max(1, M * N)

                            if density >= DENSE_BLOCK_DENSITY_THRESHOLD:
                                pair_ids = [(si, sj) for si, sj, _, _ in pair_quads]
                                r_off = np.array([starts[ri] - a_base for _, _, ri, _ in pair_quads], dtype=np.int32)
                                c_off = np.array([starts[cj] - b_base for _, _, _, cj in pair_quads], dtype=np.int32)
                                submit_gpu_job(A_ten, B_ten, pair_ids, r_off, r_len, c_off, c_len)
                            else:
                                sparse_pairs.extend(pair_quads)

                        if sparse_pairs and (active_trial is None or active_trial.can_submit()):
                            # Aggregate sparse pairs across the whole tile so no row is shattered
                            # across separate (bi, bj) block pairs, orienting the higher-degree
                            # endpoint as the row sequence.
                            s_deg = {}
                            for si, sj, _, _ in sparse_pairs:
                                s_deg[si] = s_deg.get(si, 0) + 1
                                s_deg[sj] = s_deg.get(sj, 0) + 1
                            by_row = {}
                            for si, sj, _, _ in sparse_pairs:
                                if (s_deg[si], int(lengths[si]), si) >= (s_deg[sj], int(lengths[sj]), sj):
                                    by_row.setdefault(si, []).append((si, sj, sj))
                                else:
                                    by_row.setdefault(sj, []).append((si, sj, si))

                            max_cols_emb = (32 * 1024 * 1024) // max(1, feature_dim * element_bytes)
                            max_slab_elems = max(
                                max_tile_len * max_tile_len,
                                min(tile_max_mat_elems, 33_554_432),
                            )
                            max_chunk_cols_cap = max(max_tile_len, min(block_res, max_cols_emb))

                            row_chunks = []
                            for ri, targets in by_row.items():
                                row_L = int(lengths[ri])
                                targets.sort(key=lambda x: (int(lengths[x[2]]), x[2]))
                                max_cols_mat = max_slab_elems // max(1, row_L)
                                max_cols = max(max_tile_len, min(max_cols_mat, max_chunk_cols_cap))
                                chunk_targets, chunk_cols = [], 0
                                for item in targets:
                                    L_j = int(lengths[item[2]])
                                    if chunk_targets and chunk_cols + L_j > max_cols:
                                        row_chunks.append((row_L, chunk_cols, ri, chunk_targets))
                                        chunk_targets, chunk_cols = [], 0
                                    chunk_targets.append(item)
                                    chunk_cols += L_j
                                if chunk_targets:
                                    row_chunks.append((row_L, chunk_cols, ri, chunk_targets))

                            # Sort by ascending row_L so chunks on the same shelf share height and
                            # adjacent DP warps stay balanced.
                            row_chunks.sort(key=lambda c: (c[0], c[1], c[2]))
                            shelf_width = max(c[1] for c in row_chunks)
                            max_slab_pairs = 1024

                            slab_chunks = []
                            slab_pairs = 0
                            shelf_y = 0
                            shelf_h = 0
                            cur_x = 0

                            def _flush_sparse_slab():
                                nonlocal slab_chunks, slab_pairs, shelf_y, shelf_h, cur_x
                                if not slab_chunks:
                                    return
                                actual_M = shelf_y + shelf_h
                                actual_N = max(x0 + c_cols for _, x0, _, c_cols, _, _ in slab_chunks)
                                submit_sparse_slab_job(
                                    slab_chunks,
                                    actual_M,
                                    actual_N,
                                    slab_pairs,
                                    max_slab_elems,
                                    min(max_chunk_cols_cap, shelf_width),
                                )
                                slab_chunks = []
                                slab_pairs = 0
                                shelf_y = 0
                                shelf_h = 0
                                cur_x = 0

                            for row_L, c_cols, ri, c_targets in row_chunks:
                                if active_trial is not None and not active_trial.can_submit():
                                    break
                                n_ct = len(c_targets)
                                if slab_chunks:
                                    if cur_x + c_cols <= shelf_width:
                                        cand_M = shelf_y + max(shelf_h, row_L)
                                    else:
                                        cand_M = (shelf_y + shelf_h) + row_L
                                    if (
                                        cand_M * shelf_width > max_slab_elems
                                        or slab_pairs + n_ct > max_slab_pairs
                                    ):
                                        _flush_sparse_slab()
                                        if active_trial is not None and not active_trial.can_submit():
                                            break
                                if cur_x > 0 and cur_x + c_cols > shelf_width:
                                    shelf_y += shelf_h
                                    shelf_h = 0
                                    cur_x = 0
                                slab_chunks.append((shelf_y, cur_x, row_L, c_cols, ri, c_targets))
                                cur_x += c_cols
                                if row_L > shelf_h:
                                    shelf_h = row_L
                                slab_pairs += n_ct
                            _flush_sparse_slab()

                        while inflight:
                            drain_one()
                        if int(d_nf.item()) != 0:
                            raise FloatingPointError(
                                "Fused GPU alignment scoring produced non-finite values."
                            )
                    finally:
                        # Drop EVERY tensor view referencing `packed` before the next tile
                        # uploads so two tiles never coexist on GPU simultaneously.
                        blk_slices.clear()
                        A_ten = B_ten = packed = None
                        if len(phase_tiles) > 1:
                            if is_oom_retry:
                                for ws in ws_list:
                                    ws.clear()
                            torch.cuda.empty_cache()

            if benchmark_trial is not None:
                warmup_n = benchmark_trial.warmup_count(tasks, 1, n_streams)
                run_phase(tasks[:warmup_n])
                results.clear()
                d_nf.zero_()
                active_trial = benchmark_trial
                benchmark_trial.start()
                run_phase(tasks)
                benchmark_trial.stop(len(tasks))
            else:
                run_phase(tasks)
    except BaseException as error:
        if clear_failed_frames_fn is not None:
            clear_failed_frames_fn(error)
        raise
    finally:
        try:
            for ws in ws_list:
                ws.clear()
            d_nf = None
            if backend is not None:
                backend.empty_cache()
            else:
                torch.cuda.empty_cache()
        finally:
            cap_stack.close()

    if result_callback is not None and results:
        result_callback(results)
        results.clear()
    return results


def can_use_fused_search(tasks, query_length, lengths, search_config, task_index):
    """Return whether all fixed-query search tasks fit the fused GPU search kernel."""
    if not tasks or search_config is None:
        return False
    mode = search_config["mode"]
    gap = float(search_config["gap"])
    is_local = mode == "local"
    if is_local:
        if float32_exact_local_bound(0.0, gap) is None:
            return False
    else:
        if float(np.float32(gap)) != gap:
            return False
    q_len = int(query_length)
    if q_len <= 0 or q_len > PACKED_MAX_PAIRS:
        return False
    max_t = max(int(lengths[task_index(t)]) for t in tasks)
    if max_t <= 0 or min(q_len, max_t) > PACKED_MAX_PAIRS or (q_len + max_t) > PACKED_MAX_PATH:
        return False
    return True


def run_fused_fixed_query_gpu_pipeline(
    tasks,
    *,
    query_embedding,
    store,
    lengths,
    device,
    lanes,
    search_config,
    precision="float32",
    progress=None,
    matrix_budget_override=None,
    task_index,
    timing=None,
    cuda_matmul_precision_ctx,
    precision_compute_dtype_fn,
    precision_element_bytes_fn,
    accelerator_memory_plan_fn,
    fixed_query_block_ids_fn,
    plan_upload_chunks_fn,
    chunk_reader_cls,
    to_normalized_accelerator_fn,
    torch_source_dtypes,
    fixed_query_chunk_bytes,
    fixed_query_min_chunk_bytes,
    fixed_query_chunks_per_run,
    fixed_query_ring_bytes,
    pins_host_memory_fn,
    clear_failed_frames_fn=None,
):
    """Execute 1-vs-N SSEARCH end-to-end on GPU with zero padding and zero score D2H."""
    kernels = align_kernels(device)
    if kernels is None:
        raise RuntimeError(f"Fused GPU search kernels unavailable on '{device}'.")
    tasks = list(tasks)
    if not tasks:
        return []

    element_bytes = precision_element_bytes_fn(precision)
    compute_dtype = precision_compute_dtype_fn(precision)
    plan = accelerator_memory_plan_fn(device, lanes=lanes, compute_element_bytes=element_bytes)
    n_streams = max(1, min(2, int(lanes)))
    query_array = np.asarray(query_embedding)
    query_length = int(query_array.shape[0])
    feature_dimension = int(store.feature_dimension)

    mode = search_config["mode"]
    is_local = 1 if mode == "local" else 0
    gap = float(search_config["gap"])
    norm_mode = search_config["norm_mode"]
    query_codes = np.array(search_config["query_codes"], dtype=np.uint8, copy=True, order="C")
    similar_table = np.array(search_config["similar_table"], dtype=np.uint8, copy=True, order="C")
    encode_target_fn = search_config["encode_target_fn"]
    make_row_fn = search_config["make_row_fn"]
    mismatched_err_fn = search_config.get("mismatched_error_fn")

    if len(query_codes) != query_length:
        if mismatched_err_fn is not None:
            raise mismatched_err_fn(tasks[0][1], (query_length, int(lengths[task_index(tasks[0])])), len(query_codes), int(lengths[task_index(tasks[0])]))
        raise ValueError("Query sequence length does not match query embedding length.")

    exact_bound = float32_exact_local_bound(0.0, gap) if is_local else None
    bound_val = float(exact_bound) if exact_bound is not None else float("inf")

    block_ids = fixed_query_block_ids_fn(store, query_array, plan, element_bytes)
    grouped = OrderedDict()
    for task in tasks:
        idx = task_index(task)
        grouped.setdefault(int(block_ids[idx]), []).append(task)

    nbytes = store.source_bytes
    chunk_bytes = min(
        fixed_query_chunk_bytes,
        max(
            fixed_query_min_chunk_bytes,
            sum(int(nbytes[idx]) for idx in {task_index(task) for task in tasks})
            // fixed_query_chunks_per_run,
        ),
    )
    block_layouts = []
    chunks = []
    for block_tasks in grouped.values():
        indices = sorted({task_index(task) for task in block_tasks})
        starts, rows = {}, 0
        for idx in indices:
            starts[idx] = rows
            rows += int(lengths[idx])
        planned = plan_upload_chunks_fn(indices, nbytes, chunk_bytes)
        block_layouts.append((starts, rows, len(planned)))
        chunks.extend(planned)

    pad_row = max(rows for _starts, rows, _count in block_layouts)
    capacity = max(used for _entries, used in chunks)
    ring_depth = max(2, min(8, fixed_query_ring_bytes // max(1, capacity)))
    buffers = None
    if pins_host_memory_fn(device):
        try:
            buffers = [
                torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
                for _ in range(ring_depth)
            ]
        except RuntimeError:
            buffers = None
    if buffers is None:
        buffers = [torch.empty(capacity, dtype=torch.uint8) for _ in range(ring_depth)]
    reader = chunk_reader_cls(store, chunks, [buf.numpy() for buf in buffers])

    # Target slab size in columns (keep GEMM buffer ~64 MiB per stream, honoring override)
    slab_bytes = 64 * 1024 * 1024
    if matrix_budget_override is not None and int(matrix_budget_override) > 0:
        slab_bytes = min(slab_bytes, max(4 * 1024 * 1024, int(matrix_budget_override)))
    max_slab_cols = max(1, slab_bytes // max(1, query_length * 4))

    results = []
    streams = [torch.cuda.Stream(device=device) for _ in range(n_streams)]
    upload_stream = torch.cuda.Stream(device=device)
    pending_release = deque()
    inflight = []
    ws_list = [{} for _ in range(n_streams)]
    tiny = torch.finfo(torch.float32).tiny
    packed = raw_device = query_tensor = d_qcodes = d_simtbl = d_nf = None

    def release_completed():
        while pending_release and pending_release[0][0].query():
            reader.release(pending_release.popleft()[1])

    def drain_one():
        ev, sid, slab_tasks, slab_lens = inflight.pop(0)
        ev.synchronize()
        ws = ws_list[sid]
        n_t = len(slab_tasks)
        scores = ws["h_s"][:n_t].numpy()
        metas = ws["h_m"][:n_t].numpy().view(np.uint64)
        # Check rare local f32 bound overflow
        overflow_mask = (is_local == 1) and bool(np.any(scores >= bound_val))
        batch_rows = []
        for k_i, (task, t_len, sc_val, m_val) in enumerate(
            zip(slab_tasks, slab_lens, scores.tolist(), metas.tolist())
        ):
            if overflow_mask and sc_val >= bound_val:
                # Exact CPU reference recompute on this rare target
                t_idx = task_index(task)
                t_emb = store.get(t_idx)
                ti = torch.nn.functional.normalize(
                    torch.as_tensor(query_array, dtype=torch.float32), p=2, dim=-1
                )
                tj = torch.nn.functional.normalize(
                    torch.as_tensor(t_emb, dtype=torch.float32), p=2, dim=-1
                )
                cos = torch.mm(ti, tj.T).clamp_(-1.0, 1.0)
                sim = torch.exp(-(1.0 - cos))
                rm = sim.mean(dim=1, keepdim=True)
                rs = sim.std(dim=1, keepdim=True, correction=0)
                cm = sim.mean(dim=0, keepdim=True)
                cs = sim.std(dim=0, keepdim=True, correction=0)
                ref_m = ((((sim - rm) / (rs + 1e-8)) + ((sim - cm) / (cs + 1e-8))) / 2.0).numpy()
                tc = encode_target_fn(task, t_len)
                raw_s, p_len, ident, sim_p, aln_p = local_score_length_identity(
                    ref_m, gap, query_codes, tc, similar_table
                )
                batch_rows.append(
                    make_row_fn(task[0], task[1], query_length, t_len, norm_mode, raw_s, p_len, ident, sim_p, aln_p)
                )
            else:
                p_len = int(m_val >> 45)
                aln_p = int((m_val >> 30) & 0x7FFF)
                sim_p = int((m_val >> 15) & 0x7FFF)
                ident = int(m_val & 0x7FFF)
                batch_rows.append(
                    make_row_fn(task[0], task[1], query_length, t_len, norm_mode, float(sc_val), p_len, ident, sim_p, aln_p)
                )
        results.extend(batch_rows)
        if progress is not None:
            progress.update(len(batch_rows))

    def idle():
        if inflight and inflight[0][0].query():
            drain_one()
        release_completed()

    def ensure_ws(ws, n_cols, n_t):
        need_mat = query_length * n_cols
        if "mat" not in ws or ws["mat"].numel() < need_mat:
            ws["mat"] = torch.empty(need_mat, dtype=torch.float32, device=device)
        need_r = n_t * query_length
        if "rm" not in ws or ws["rm"].shape[0] < need_r:
            sz = max(need_r, 4096)
            ws["rm"] = torch.empty(sz, dtype=torch.float32, device=device)
            ws["rs"] = torch.empty(sz, dtype=torch.float32, device=device)
            ws["sc_s"] = torch.empty(sz, dtype=torch.float32, device=device)
            ws["sc_m"] = torch.empty(sz, dtype=torch.int64, device=device)
        if "cm" not in ws or ws["cm"].shape[0] < n_cols:
            sz = max(n_cols, 4096)
            ws["cm"] = torch.empty(sz, dtype=torch.float32, device=device)
            ws["cs"] = torch.empty(sz, dtype=torch.float32, device=device)
        if "os" not in ws or ws["os"].shape[0] < n_t:
            sz = max(n_t, 512)
            ws["os"] = torch.empty(sz, dtype=torch.float32, device=device)
            ws["om"] = torch.empty(sz, dtype=torch.int64, device=device)
            ws["h_s"] = torch.empty(sz, dtype=torch.float32, pin_memory=True)
            ws["h_m"] = torch.empty(sz, dtype=torch.int64, pin_memory=True)

    def upload_block(layout, after):
        starts, _rows, chunk_count = layout
        with torch.cuda.stream(upload_stream):
            waited = False
            for _ in range(chunk_count):
                _number, entries, used, slot = reader.next_chunk(idle)
                raw_device[:used].copy_(buffers[slot][:used], non_blocking=True)
                copied = torch.cuda.Event()
                copied.record(upload_stream)
                pending_release.append((copied, slot))
                if not waited:
                    for ev in after:
                        upload_stream.wait_event(ev)
                    waited = True
                for index, offset, size in entries:
                    length = int(lengths[index])
                    source = raw_device[offset:offset + size].view(
                        torch_source_dtypes[store.dtypes[index]]
                    ).view(length, feature_dimension)
                    values = source.to(torch.float32)
                    norms = torch.linalg.vector_norm(values, ord=2, dim=-1, keepdim=True)
                    norms.clamp_min_(tiny)
                    target = packed[starts[index]:starts[index] + length]
                    if compute_dtype == torch.float32:
                        torch.div(values, norms, out=target)
                    else:
                        target.copy_(torch.div(values, norms))
                release_completed()
            uploaded = torch.cuda.Event()
            uploaded.record(upload_stream)
        return uploaded

    try:
        with cuda_matmul_precision_ctx(precision), torch.inference_mode():
            d_qcodes = torch.from_numpy(query_codes).to(device=device)
            d_simtbl = torch.from_numpy(similar_table.reshape(-1)).to(device=device)
            d_nf = torch.zeros(1, dtype=torch.int32, device=device)
            with torch.cuda.stream(streams[0]):
                query_tensor = to_normalized_accelerator_fn(query_array, device, precision)
                q_ev = torch.cuda.Event()
                q_ev.record(streams[0])
            q_ev.synchronize()
            with torch.cuda.stream(upload_stream):
                packed = torch.empty((max(1, pad_row), feature_dimension), dtype=compute_dtype, device=device)
                raw_device = torch.empty(capacity, dtype=torch.uint8, device=device)

            if timing is not None:
                timing.start()
            reader.start()

            previous_lanes = []
            stream_cursor = 0
            for layout, block_tasks in zip(block_layouts, grouped.values()):
                starts = layout[0]
                uploaded = upload_block(layout, previous_lanes)
                for st in streams:
                    st.wait_event(uploaded)

                # Order block_tasks by packed start offset so each slab is a single contiguous slice!
                ordered_tasks = sorted(block_tasks, key=lambda t: starts[task_index(t)])
                # Also handle any duplicate target indices gracefully by slicing exact views if non-contiguous
                cursor = 0
                n_bt = len(ordered_tasks)
                while cursor < n_bt:
                    slab_tasks = []
                    slab_lens = []
                    slab_tc_parts = []
                    slab_cols = 0
                    contiguous = True
                    first_start = starts[task_index(ordered_tasks[cursor])]
                    expected_next = first_start
                    while cursor < n_bt:
                        t = ordered_tasks[cursor]
                        t_idx = task_index(t)
                        t_len = int(lengths[t_idx])
                        if slab_tasks and slab_cols + t_len > max_slab_cols:
                            break
                        tc = encode_target_fn(t, t_len)
                        if len(tc) != t_len:
                            if mismatched_err_fn is not None:
                                raise mismatched_err_fn(t[1], (query_length, t_len), len(query_codes), len(tc))
                            raise ValueError("Target sequence length does not match embedding length.")
                        if starts[t_idx] != expected_next:
                            contiguous = False
                        expected_next = starts[t_idx] + t_len
                        slab_tasks.append(t)
                        slab_lens.append(t_len)
                        slab_tc_parts.append(tc)
                        slab_cols += t_len
                        cursor += 1

                    sid = stream_cursor % n_streams
                    stream_cursor += 1
                    while len(inflight) >= n_streams:
                        drain_one()
                    st = streams[sid]
                    ws = ws_list[sid]
                    n_t = len(slab_tasks)
                    cl_np = np.array(slab_lens, dtype=np.int32)
                    co_np = np.zeros(n_t, dtype=np.int32)
                    if n_t > 1:
                        co_np[1:] = np.cumsum(cl_np[:-1], dtype=np.int32)
                    ro_np = np.zeros(n_t, dtype=np.int32)
                    rl_np = np.full(n_t, query_length, dtype=np.int32)
                    rs_np = (np.arange(n_t, dtype=np.int32) * query_length)
                    meta_h = torch.from_numpy(np.stack([ro_np, rl_np, co_np, cl_np, rs_np, co_np], axis=0))
                    tc_h = torch.from_numpy(np.concatenate(slab_tc_parts))
                    try:
                        meta_h = meta_h.pin_memory()
                        tc_h = tc_h.pin_memory()
                        non_block = True
                    except RuntimeError:
                        non_block = False

                    with torch.cuda.stream(st):
                        ensure_ws(ws, slab_cols, n_t)
                        # Pin host and device metadata buffers in the per-stream workspace until
                        # drain_one() synchronizes `st`, since raw cuLaunchKernel does not register
                        # tensor lifetimes with PyTorch's caching allocator.
                        ws["meta_h"] = meta_h
                        ws["tc_h"] = tc_h
                        meta_d = meta_h.to(device=device, non_blocking=non_block)
                        tc_d = tc_h.to(device=device, non_blocking=non_block)
                        ws["meta_d"] = meta_d
                        ws["tc_d"] = tc_d
                        if contiguous:
                            B_ten = packed[first_start:first_start + slab_cols]
                        else:
                            B_ten = torch.cat(
                                [
                                    packed[starts[task_index(t)]:starts[task_index(t)] + L]
                                    for t, L in zip(slab_tasks, slab_lens)
                                ],
                                dim=0,
                            )
                        ws["B_ten"] = B_ten
                        mat = ws["mat"][:query_length * slab_cols].view(query_length, slab_cols)
                        if compute_dtype == torch.float32:
                            torch.mm(query_tensor, B_ten.T, out=mat)
                        else:
                            mat.copy_(torch.mm(query_tensor, B_ten.T).to(torch.float32))
                        kernels.launch(
                            "align_exp_stats",
                            (n_t, 1, 1),
                            (128, 1, 1),
                            [
                                mat, slab_cols, meta_d[0], meta_d[1], meta_d[2], meta_d[3],
                                meta_d[4], meta_d[5], ws["rm"], ws["rs"], ws["cm"], ws["cs"],
                                d_nf, n_t,
                            ],
                            st.cuda_stream,
                        )
                        kernels.launch(
                            "search_dp_k4",
                            ((n_t + 3) // 4, 1, 1),
                            (128, 1, 1),
                            [
                                mat, slab_cols, meta_d[2], meta_d[3], meta_d[4], meta_d[5],
                                ws["rm"], ws["rs"], ws["cm"], ws["cs"],
                                d_qcodes, tc_d, d_simtbl, ws["sc_s"], ws["sc_m"], query_length,
                                ws["os"], ws["om"], query_length, n_t, is_local, gap, 2.0,
                            ],
                            st.cuda_stream,
                        )
                        ws["h_s"][:n_t].copy_(ws["os"][:n_t], non_blocking=True)
                        ws["h_m"][:n_t].copy_(ws["om"][:n_t], non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record(st)
                    inflight.append((ev, sid, slab_tasks, slab_lens))
                    if inflight and inflight[0][0].query():
                        drain_one()
                    release_completed()

                previous_lanes = []
                for st in streams:
                    reached = torch.cuda.Event()
                    reached.record(st)
                    previous_lanes.append(reached)

            while inflight:
                drain_one()
            if int(d_nf.item()) != 0:
                raise FloatingPointError(
                    "Fixed-query accelerator scoring produced non-finite values."
                )
            if timing is not None:
                timing.stop()
    except BaseException as error:
        if clear_failed_frames_fn is not None:
            clear_failed_frames_fn(error)
        raise
    finally:
        reader.close()
        inflight.clear()
        pending_release.clear()
        try:
            torch.cuda.synchronize(device)
        except RuntimeError:
            pass
        for ws in ws_list:
            ws.clear()
        packed = raw_device = query_tensor = d_qcodes = d_simtbl = d_nf = None
        torch.cuda.empty_cache()

    if timing is not None:
        timing.finish()
    return results


__all__ = [
    "AlignKernels",
    "DISABLE_ENVIRONMENT_VARIABLE",
    "KERNEL_NAMES",
    "KERNEL_SOURCE",
    "align_kernels",
    "can_use_fused_search",
    "fallback_reason",
    "kernels_disabled",
    "run_fused_fixed_query_gpu_pipeline",
    "run_fused_gpu_alignment_pipeline",
]
