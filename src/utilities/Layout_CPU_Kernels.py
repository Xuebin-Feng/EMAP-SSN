# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Disk-cached Numba CPU physics kernels for Layout_Engine_SSN.

Keeping all CPU layout ``@jit(..., cache=True)`` kernels in this canonical
``utilities.Layout_CPU_Kernels`` module eliminates multi-second LLVM
recompilation on every subprocess launch while preserving bit-for-bit
equivalence with the serial all-pairs physics formulas in 2D and 3D.
"""

from __future__ import annotations

import math
import numpy as np

try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    NUMBA_AVAILABLE = False
    prange = range


# Every fastmath flag except 'contract' so Euler updates keep separate
# multiply and add roundings (matching slice arithmetic bit-for-bit).
_UNFUSED_FASTMATH = {"nnan", "ninf", "nsz", "arcp", "afn", "reassoc"}


def _compile_cached(func=None, *, parallel=False, fastmath=True):
    def decorator(fn):
        if NUMBA_AVAILABLE:
            return jit(nopython=True, fastmath=fastmath, parallel=parallel, cache=True)(fn)
        return fn
    return decorator(func) if func is not None else decorator


@_compile_cached
def group_active_nodes(comp_labels, active_nodes):
    """Group active nodes by component, preserving order within each component."""
    n_active = active_nodes.shape[0]
    order = np.argsort(comp_labels[active_nodes], kind="mergesort")
    members = active_nodes[order]
    run_starts = np.empty(n_active, dtype=np.int64)
    run_ends = np.empty(n_active, dtype=np.int64)
    run_start = 0
    while run_start < n_active:
        label = comp_labels[members[run_start]]
        run_end = run_start + 1
        while run_end < n_active and comp_labels[members[run_end]] == label:
            run_end += 1
        for position in range(run_start, run_end):
            run_starts[position] = run_start
            run_ends[position] = run_end
        run_start = run_end
    return members, run_starts, run_ends


@_compile_cached
def sort_int32_inplace(arr, n):
    """In-place Shell sort for small-to-medium int32 neighbor hit lists."""
    if n <= 1:
        return
    gap = 1
    while gap < n // 3:
        gap = gap * 3 + 1
    while gap > 0:
        for i in range(gap, n):
            key = arr[i]
            j = i - gap
            while j >= 0 and arr[j] > key:
                arr[j + gap] = arr[j]
                j -= gap
            arr[j + gap] = key
        gap //= 3


# ---------------------------------------------------------------------------
# 2D Kernels
# ---------------------------------------------------------------------------

@_compile_cached(parallel=True)
def repulsion_grid_run_2d(
    run_pos, members, acc, repulsion, r0, m,
    cell_cx, cell_cy, ny, nx, cell_ptr, cell_nodes, cell_pos, hit_bufs,
    k_coul, max_f, max_total_repulsion, cutoff_dist, cutoff_sq, taper_start, taper_width,
):
    for local in prange(m):
        position = r0 + local
        node = members[position]
        px = run_pos[position, 0]
        py = run_pos[position, 1]
        cx0 = cell_cx[local]
        cy0 = cell_cy[local]

        tid = numba.get_thread_id() if NUMBA_AVAILABLE else 0
        hits = hit_bufs[tid]
        n_hits = 0

        cx_lo = cx0 - 1 if cx0 > 0 else 0
        cx_hi = cx0 + 1 if cx0 + 1 < nx else nx - 1
        cy_lo = cy0 - 1 if cy0 > 0 else 0
        cy_hi = cy0 + 1 if cy0 + 1 < ny else ny - 1

        slice_total = 0
        for cx in range(cx_lo, cx_hi + 1):
            slice_total += cell_ptr[cx * ny + cy_hi + 1] - cell_ptr[cx * ny + cy_lo]

        if slice_total * 5 > m:
            for other_pos in range(r0, r0 + m):
                if other_pos == position:
                    continue
                dx = px - run_pos[other_pos, 0]
                dy = py - run_pos[other_pos, 1]
                dist_sq = dx * dx + dy * dy
                if dist_sq > cutoff_sq or dist_sq == 0.0:
                    continue
                dist = math.sqrt(dist_sq)
                safe_dist = max(dist, 0.5)
                f = k_coul / (safe_dist ** 2)
                if max_f > 0.0 and f > max_f:
                    f = max_f
                if dist > taper_start:
                    f *= max(0.0, (cutoff_dist - dist) / taper_width)
                repulsion[node, 0] += f * (dx / dist)
                repulsion[node, 1] += f * (dy / dist)
        else:
            for cx in range(cx_lo, cx_hi + 1):
                s0 = cell_ptr[cx * ny + cy_lo]
                s1 = cell_ptr[cx * ny + cy_hi + 1]
                for s in range(s0, s1):
                    dx = px - cell_pos[s, 0]
                    dy = py - cell_pos[s, 1]
                    dist_sq = dx * dx + dy * dy
                    if dist_sq <= cutoff_sq and dist_sq > 0.0:
                        other_local = cell_nodes[s]
                        if other_local != local:
                            hits[n_hits] = other_local
                            n_hits += 1

            sort_int32_inplace(hits, n_hits)

            for h_i in range(n_hits):
                other_pos = r0 + hits[h_i]
                dx = px - run_pos[other_pos, 0]
                dy = py - run_pos[other_pos, 1]
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq)
                safe_dist = max(dist, 0.5)
                f = k_coul / (safe_dist ** 2)
                if max_f > 0.0 and f > max_f:
                    f = max_f
                if dist > taper_start:
                    f *= max(0.0, (cutoff_dist - dist) / taper_width)
                repulsion[node, 0] += f * (dx / dist)
                repulsion[node, 1] += f * (dy / dist)

        rep_norm = math.sqrt(
            repulsion[node, 0] * repulsion[node, 0]
            + repulsion[node, 1] * repulsion[node, 1]
        )
        if max_total_repulsion > 0.0 and rep_norm > max_total_repulsion:
            rep_scale = max_total_repulsion / rep_norm
            repulsion[node, 0] *= rep_scale
            repulsion[node, 1] *= rep_scale
        acc[node, 0] += repulsion[node, 0]
        acc[node, 1] += repulsion[node, 1]


@_compile_cached(parallel=True)
def repulsion_direct_range_2d(
    run_pos, members, run_starts, run_ends, acc, repulsion, lo, hi,
    k_coul, max_f, max_total_repulsion, cutoff_dist, cutoff_sq, taper_start, taper_width,
):
    for position in prange(lo, hi):
        node = members[position]
        px = run_pos[position, 0]
        py = run_pos[position, 1]
        for other_position in range(run_starts[position], run_ends[position]):
            if other_position == position:
                continue

            dx = px - run_pos[other_position, 0]
            dy = py - run_pos[other_position, 1]
            dist_sq = dx * dx + dy * dy

            if dist_sq > cutoff_sq:
                continue
            if dist_sq == 0.0:
                continue

            dist = math.sqrt(dist_sq)
            safe_dist = max(dist, 0.5)
            f = k_coul / (safe_dist ** 2)
            if max_f > 0.0 and f > max_f:
                f = max_f
            if dist > taper_start:
                f *= max(0.0, (cutoff_dist - dist) / taper_width)

            repulsion[node, 0] += f * (dx / dist)
            repulsion[node, 1] += f * (dy / dist)

        rep_norm = math.sqrt(
            repulsion[node, 0] * repulsion[node, 0]
            + repulsion[node, 1] * repulsion[node, 1]
        )
        if max_total_repulsion > 0.0 and rep_norm > max_total_repulsion:
            rep_scale = max_total_repulsion / rep_norm
            repulsion[node, 0] *= rep_scale
            repulsion[node, 1] *= rep_scale
        acc[node, 0] += repulsion[node, 0]
        acc[node, 1] += repulsion[node, 1]


@_compile_cached
def accumulate_repulsion_grouped_2d(
    pos, acc, repulsion, run_pos, members, run_starts, run_ends, n_threads,
    k_coul, max_f, max_total_repulsion, cutoff_dist,
):
    cutoff_sq = cutoff_dist * cutoff_dist
    taper_start = cutoff_dist * 0.8
    taper_width = max(cutoff_dist * 0.2, 1e-9)
    n_active = members.shape[0]
    if n_active == 0:
        return

    for position in range(n_active):
        node = members[position]
        run_pos[position, 0] = pos[node, 0]
        run_pos[position, 1] = pos[node, 1]

    has_large = False
    p = 0
    while p < n_active:
        end = run_ends[p]
        if end - p >= 512:
            has_large = True
            break
        p = end

    if not has_large or cutoff_dist <= 0.0 or not math.isfinite(cutoff_dist):
        repulsion_direct_range_2d(
            run_pos, members, run_starts, run_ends, acc, repulsion, 0, n_active,
            k_coul, max_f, max_total_repulsion, cutoff_dist, cutoff_sq, taper_start, taper_width,
        )
        return

    h = cutoff_dist * 1.0001
    inv_h = 1.0 / h
    p = 0
    direct_lo = -1
    while p < n_active:
        end = run_ends[p]
        m = end - p
        used_grid = False
        if m >= 512:
            min_x = float(run_pos[p, 0]); max_x = min_x
            min_y = float(run_pos[p, 1]); max_y = min_y
            for idx in range(p + 1, end):
                vx = float(run_pos[idx, 0])
                vy = float(run_pos[idx, 1])
                if vx < min_x: min_x = vx
                elif vx > max_x: max_x = vx
                if vy < min_y: min_y = vy
                elif vy > max_y: max_y = vy
            span_x = max_x - min_x
            span_y = max_y - min_y
            if (
                math.isfinite(span_x) and math.isfinite(span_y)
                and span_x * inv_h < 1e4 and span_y * inv_h < 1e4
            ):
                nx = int(math.floor(span_x * inv_h)) + 1
                ny = int(math.floor(span_y * inv_h)) + 1
                n_cells = nx * ny
                if 36 <= n_cells <= 262144:
                    cell_cx = np.empty(m, dtype=np.int32)
                    cell_cy = np.empty(m, dtype=np.int32)
                    counts = np.zeros(n_cells, dtype=np.int32)
                    for local in range(m):
                        cx = int(math.floor((float(run_pos[p + local, 0]) - min_x) * inv_h))
                        cy = int(math.floor((float(run_pos[p + local, 1]) - min_y) * inv_h))
                        cell_cx[local] = cx
                        cell_cy[local] = cy
                        counts[cx * ny + cy] += 1
                    cand_pairs = np.int64(0)
                    for cx_i in range(nx):
                        cx_lo = cx_i - 1 if cx_i > 0 else 0
                        cx_hi = cx_i + 1 if cx_i + 1 < nx else nx - 1
                        row_off = cx_i * ny
                        for cy_i in range(ny):
                            c_here = counts[row_off + cy_i]
                            if c_here == 0:
                                continue
                            cy_lo = cy_i - 1 if cy_i > 0 else 0
                            cy_hi = cy_i + 1 if cy_i + 1 < ny else ny - 1
                            nb = np.int64(0)
                            for nx_i in range(cx_lo, cx_hi + 1):
                                n_off = nx_i * ny
                                for ny_i in range(cy_lo, cy_hi + 1):
                                    nb += counts[n_off + ny_i]
                            cand_pairs += np.int64(c_here) * nb
                    if cand_pairs * 7 < np.int64(m) * np.int64(m):
                        cell_ptr = np.empty(n_cells + 1, dtype=np.int32)
                        acc_cnt = 0
                        for cid in range(n_cells):
                            cell_ptr[cid] = acc_cnt
                            acc_cnt += counts[cid]
                        cell_ptr[n_cells] = acc_cnt
                        fill = cell_ptr[:-1].copy()
                        cell_nodes = np.empty(m, dtype=np.int32)
                        cell_pos = np.empty((m, 2), dtype=pos.dtype)
                        for local in range(m):
                            cid = cell_cx[local] * ny + cell_cy[local]
                            dst = fill[cid]
                            fill[cid] = dst + 1
                            cell_nodes[dst] = local
                            cell_pos[dst, 0] = run_pos[p + local, 0]
                            cell_pos[dst, 1] = run_pos[p + local, 1]
                        hit_bufs = np.empty((n_threads, m), dtype=np.int32)
                        if direct_lo != -1:
                            repulsion_direct_range_2d(
                                run_pos, members, run_starts, run_ends, acc, repulsion, direct_lo, p,
                                k_coul, max_f, max_total_repulsion, cutoff_dist, cutoff_sq, taper_start, taper_width,
                            )
                            direct_lo = -1
                        repulsion_grid_run_2d(
                            run_pos, members, acc, repulsion, p, m,
                            cell_cx, cell_cy, ny, nx, cell_ptr, cell_nodes, cell_pos, hit_bufs,
                            k_coul, max_f, max_total_repulsion, cutoff_dist, cutoff_sq, taper_start, taper_width,
                        )
                        used_grid = True
        if not used_grid and direct_lo == -1:
            direct_lo = p
        p = end

    if direct_lo != -1:
        repulsion_direct_range_2d(
            run_pos, members, run_starts, run_ends, acc, repulsion, direct_lo, n_active,
            k_coul, max_f, max_total_repulsion, cutoff_dist, cutoff_sq, taper_start, taper_width,
        )


@_compile_cached(fastmath=_UNFUSED_FASTMATH)
def euler_update_2d(pos, vel, acc, i, dt, damping):
    acc[i, 0] -= damping * vel[i, 0]; acc[i, 1] -= damping * vel[i, 1]
    vel[i, 0] += acc[i, 0] * dt; vel[i, 1] += acc[i, 1] * dt
    pos[i, 0] += vel[i, 0] * dt; pos[i, 1] += vel[i, 1] * dt


@_compile_cached
def run_physics_kernel_precomputed_2d(
    pos, vel, springs, active_mask, active_nodes, box_limits,
    members, run_starts, run_ends, acc, repulsion, run_pos, n_threads,
    dt, damping, k_spr, k_coul, max_f, max_total_repulsion, cutoff_dist,
):
    n_balls = pos.shape[0]
    for i in range(n_balls):
        acc[i, 0] = 0.0
        acc[i, 1] = 0.0
        repulsion[i, 0] = 0.0
        repulsion[i, 1] = 0.0

    for i in range(springs.shape[0]):
        idx_a, idx_b = springs[i, 0], springs[i, 1]
        if not active_mask[idx_a] or not active_mask[idx_b]:
            continue
        dx, dy = pos[idx_a, 0] - pos[idx_b, 0], pos[idx_a, 1] - pos[idx_b, 1]
        dist = math.sqrt(dx * dx + dy * dy) + 1e-9
        f = -k_spr * dist
        acc[idx_a, 0] += f * (dx / dist); acc[idx_a, 1] += f * (dy / dist)
        acc[idx_b, 0] -= f * (dx / dist); acc[idx_b, 1] -= f * (dy / dist)

    accumulate_repulsion_grouped_2d(
        pos, acc, repulsion, run_pos, members, run_starts, run_ends, n_threads,
        k_coul, max_f, max_total_repulsion, cutoff_dist,
    )

    rmsd = 0.0
    n_active = active_nodes.shape[0]
    for i in range(n_balls):
        if not active_mask[i]:
            continue
        box_limit = box_limits[i]
        old_x, old_y = pos[i, 0], pos[i, 1]
        euler_update_2d(pos, vel, acc, i, dt, damping)

        if pos[i, 0] > box_limit: pos[i, 0] = box_limit; vel[i, 0] *= -0.5
        elif pos[i, 0] < -box_limit: pos[i, 0] = -box_limit; vel[i, 0] *= -0.5
        if pos[i, 1] > box_limit: pos[i, 1] = box_limit; vel[i, 1] *= -0.5
        elif pos[i, 1] < -box_limit: pos[i, 1] = -box_limit; vel[i, 1] *= -0.5

        diff_x, diff_y = pos[i, 0] - old_x, pos[i, 1] - old_y
        rmsd += diff_x ** 2 + diff_y ** 2

    if n_active == 0:
        return 0.0
    return math.sqrt(rmsd / n_active)


# ---------------------------------------------------------------------------
# 3D Kernels
# ---------------------------------------------------------------------------

@_compile_cached(parallel=True)
def repulsion_grid_run_3d(
    run_pos, members, acc, repulsion, r0, m,
    cell_cx, cell_cy, cell_cz, nx, ny, nz, cell_ptr, cell_nodes, cell_pos, hit_bufs,
    k_coul, max_f, max_total_repulsion, cutoff_dist, cutoff_sq, taper_start, taper_width,
):
    for local in prange(m):
        position = r0 + local
        node = members[position]
        px = run_pos[position, 0]
        py = run_pos[position, 1]
        pz = run_pos[position, 2]
        cx0 = cell_cx[local]
        cy0 = cell_cy[local]
        cz0 = cell_cz[local]

        tid = numba.get_thread_id() if NUMBA_AVAILABLE else 0
        hits = hit_bufs[tid]
        n_hits = 0

        cx_lo = cx0 - 1 if cx0 > 0 else 0
        cx_hi = cx0 + 1 if cx0 + 1 < nx else nx - 1
        cy_lo = cy0 - 1 if cy0 > 0 else 0
        cy_hi = cy0 + 1 if cy0 + 1 < ny else ny - 1
        cz_lo = cz0 - 1 if cz0 > 0 else 0
        cz_hi = cz0 + 1 if cz0 + 1 < nz else nz - 1

        slice_total = 0
        for cx in range(cx_lo, cx_hi + 1):
            cxy_base = cx * ny
            for cy in range(cy_lo, cy_hi + 1):
                row_base = (cxy_base + cy) * nz
                slice_total += cell_ptr[row_base + cz_hi + 1] - cell_ptr[row_base + cz_lo]

        if slice_total * 5 > m:
            for other_pos in range(r0, r0 + m):
                if other_pos == position:
                    continue
                dx = px - run_pos[other_pos, 0]
                dy = py - run_pos[other_pos, 1]
                dz = pz - run_pos[other_pos, 2]
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq > cutoff_sq or dist_sq == 0.0:
                    continue
                dist = math.sqrt(dist_sq)
                safe_dist = max(dist, 0.5)
                f = k_coul / (safe_dist ** 2)
                if max_f > 0.0 and f > max_f:
                    f = max_f
                if dist > taper_start:
                    f *= max(0.0, (cutoff_dist - dist) / taper_width)
                repulsion[node, 0] += f * (dx / dist)
                repulsion[node, 1] += f * (dy / dist)
                repulsion[node, 2] += f * (dz / dist)
        else:
            for cx in range(cx_lo, cx_hi + 1):
                cxy_base = cx * ny
                for cy in range(cy_lo, cy_hi + 1):
                    row_base = (cxy_base + cy) * nz
                    s0 = cell_ptr[row_base + cz_lo]
                    s1 = cell_ptr[row_base + cz_hi + 1]
                    for s in range(s0, s1):
                        dx = px - cell_pos[s, 0]
                        dy = py - cell_pos[s, 1]
                        dz = pz - cell_pos[s, 2]
                        dist_sq = dx * dx + dy * dy + dz * dz
                        if dist_sq <= cutoff_sq and dist_sq > 0.0:
                            other_local = cell_nodes[s]
                            if other_local != local:
                                hits[n_hits] = other_local
                                n_hits += 1

            sort_int32_inplace(hits, n_hits)

            for h_i in range(n_hits):
                other_pos = r0 + hits[h_i]
                dx = px - run_pos[other_pos, 0]
                dy = py - run_pos[other_pos, 1]
                dz = pz - run_pos[other_pos, 2]
                dist_sq = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(dist_sq)
                safe_dist = max(dist, 0.5)
                f = k_coul / (safe_dist ** 2)
                if max_f > 0.0 and f > max_f:
                    f = max_f
                if dist > taper_start:
                    f *= max(0.0, (cutoff_dist - dist) / taper_width)
                repulsion[node, 0] += f * (dx / dist)
                repulsion[node, 1] += f * (dy / dist)
                repulsion[node, 2] += f * (dz / dist)

        rep_norm = math.sqrt(
            repulsion[node, 0] * repulsion[node, 0]
            + repulsion[node, 1] * repulsion[node, 1]
            + repulsion[node, 2] * repulsion[node, 2]
        )
        if max_total_repulsion > 0.0 and rep_norm > max_total_repulsion:
            rep_scale = max_total_repulsion / rep_norm
            repulsion[node, 0] *= rep_scale
            repulsion[node, 1] *= rep_scale
            repulsion[node, 2] *= rep_scale
        acc[node, 0] += repulsion[node, 0]
        acc[node, 1] += repulsion[node, 1]
        acc[node, 2] += repulsion[node, 2]


@_compile_cached(parallel=True)
def repulsion_direct_range_3d(
    run_pos, members, run_starts, run_ends, acc, repulsion, lo, hi,
    k_coul, max_f, max_total_repulsion, cutoff_dist, cutoff_sq, taper_start, taper_width,
):
    for position in prange(lo, hi):
        node = members[position]
        px = run_pos[position, 0]
        py = run_pos[position, 1]
        pz = run_pos[position, 2]
        for other_position in range(run_starts[position], run_ends[position]):
            if other_position == position:
                continue

            dx = px - run_pos[other_position, 0]
            dy = py - run_pos[other_position, 1]
            dz = pz - run_pos[other_position, 2]
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq > cutoff_sq:
                continue
            if dist_sq == 0.0:
                continue

            dist = math.sqrt(dist_sq)
            safe_dist = max(dist, 0.5)
            f = k_coul / (safe_dist ** 2)
            if max_f > 0.0 and f > max_f:
                f = max_f
            if dist > taper_start:
                f *= max(0.0, (cutoff_dist - dist) / taper_width)

            repulsion[node, 0] += f * (dx / dist)
            repulsion[node, 1] += f * (dy / dist)
            repulsion[node, 2] += f * (dz / dist)

        rep_norm = math.sqrt(
            repulsion[node, 0] * repulsion[node, 0]
            + repulsion[node, 1] * repulsion[node, 1]
            + repulsion[node, 2] * repulsion[node, 2]
        )
        if max_total_repulsion > 0.0 and rep_norm > max_total_repulsion:
            rep_scale = max_total_repulsion / rep_norm
            repulsion[node, 0] *= rep_scale
            repulsion[node, 1] *= rep_scale
            repulsion[node, 2] *= rep_scale
        acc[node, 0] += repulsion[node, 0]
        acc[node, 1] += repulsion[node, 1]
        acc[node, 2] += repulsion[node, 2]


@_compile_cached
def accumulate_repulsion_grouped_3d(
    pos, acc, repulsion, run_pos, members, run_starts, run_ends, n_threads,
    k_coul, max_f, max_total_repulsion, cutoff_dist,
):
    cutoff_sq = cutoff_dist * cutoff_dist
    taper_start = cutoff_dist * 0.8
    taper_width = max(cutoff_dist * 0.2, 1e-9)
    n_active = members.shape[0]
    if n_active == 0:
        return

    for position in range(n_active):
        node = members[position]
        run_pos[position, 0] = pos[node, 0]
        run_pos[position, 1] = pos[node, 1]
        run_pos[position, 2] = pos[node, 2]

    has_large = False
    p = 0
    while p < n_active:
        end = run_ends[p]
        if end - p >= 512:
            has_large = True
            break
        p = end

    if not has_large or cutoff_dist <= 0.0 or not math.isfinite(cutoff_dist):
        repulsion_direct_range_3d(
            run_pos, members, run_starts, run_ends, acc, repulsion, 0, n_active,
            k_coul, max_f, max_total_repulsion, cutoff_dist, cutoff_sq, taper_start, taper_width,
        )
        return

    h = cutoff_dist * 1.0001
    inv_h = 1.0 / h
    p = 0
    direct_lo = -1
    while p < n_active:
        end = run_ends[p]
        m = end - p
        used_grid = False
        if m >= 512:
            min_x = float(run_pos[p, 0]); max_x = min_x
            min_y = float(run_pos[p, 1]); max_y = min_y
            min_z = float(run_pos[p, 2]); max_z = min_z
            for idx in range(p + 1, end):
                vx = float(run_pos[idx, 0])
                vy = float(run_pos[idx, 1])
                vz = float(run_pos[idx, 2])
                if vx < min_x: min_x = vx
                elif vx > max_x: max_x = vx
                if vy < min_y: min_y = vy
                elif vy > max_y: max_y = vy
                if vz < min_z: min_z = vz
                elif vz > max_z: max_z = vz
            span_x = max_x - min_x
            span_y = max_y - min_y
            span_z = max_z - min_z
            if (
                math.isfinite(span_x) and math.isfinite(span_y) and math.isfinite(span_z)
                and span_x * inv_h < 256.0 and span_y * inv_h < 256.0 and span_z * inv_h < 256.0
            ):
                nx = int(math.floor(span_x * inv_h)) + 1
                ny = int(math.floor(span_y * inv_h)) + 1
                nz = int(math.floor(span_z * inv_h)) + 1
                n_cells = nx * ny * nz
                if 125 <= n_cells <= 262144:
                    cell_cx = np.empty(m, dtype=np.int32)
                    cell_cy = np.empty(m, dtype=np.int32)
                    cell_cz = np.empty(m, dtype=np.int32)
                    counts = np.zeros(n_cells, dtype=np.int32)
                    for local in range(m):
                        cx = int(math.floor((float(run_pos[p + local, 0]) - min_x) * inv_h))
                        cy = int(math.floor((float(run_pos[p + local, 1]) - min_y) * inv_h))
                        cz = int(math.floor((float(run_pos[p + local, 2]) - min_z) * inv_h))
                        cell_cx[local] = cx
                        cell_cy[local] = cy
                        cell_cz[local] = cz
                        counts[(cx * ny + cy) * nz + cz] += 1
                    cand_pairs = np.int64(0)
                    for cx_i in range(nx):
                        cx_lo = cx_i - 1 if cx_i > 0 else 0
                        cx_hi = cx_i + 1 if cx_i + 1 < nx else nx - 1
                        for cy_i in range(ny):
                            cy_lo = cy_i - 1 if cy_i > 0 else 0
                            cy_hi = cy_i + 1 if cy_i + 1 < ny else ny - 1
                            base_xy = (cx_i * ny + cy_i) * nz
                            for cz_i in range(nz):
                                c_here = counts[base_xy + cz_i]
                                if c_here == 0:
                                    continue
                                cz_lo = cz_i - 1 if cz_i > 0 else 0
                                cz_hi = cz_i + 1 if cz_i + 1 < nz else nz - 1
                                nb = np.int64(0)
                                for nx_i in range(cx_lo, cx_hi + 1):
                                    for ny_i in range(cy_lo, cy_hi + 1):
                                        n_base = (nx_i * ny + ny_i) * nz
                                        for nz_i in range(cz_lo, cz_hi + 1):
                                            nb += counts[n_base + nz_i]
                                cand_pairs += np.int64(c_here) * nb
                    if cand_pairs * 7 < np.int64(m) * np.int64(m):
                        cell_ptr = np.empty(n_cells + 1, dtype=np.int32)
                        acc_cnt = 0
                        for cid in range(n_cells):
                            cell_ptr[cid] = acc_cnt
                            acc_cnt += counts[cid]
                        cell_ptr[n_cells] = acc_cnt
                        fill = cell_ptr[:-1].copy()
                        cell_nodes = np.empty(m, dtype=np.int32)
                        cell_pos = np.empty((m, 3), dtype=pos.dtype)
                        for local in range(m):
                            cid = (cell_cx[local] * ny + cell_cy[local]) * nz + cell_cz[local]
                            dst = fill[cid]
                            fill[cid] = dst + 1
                            cell_nodes[dst] = local
                            cell_pos[dst, 0] = run_pos[p + local, 0]
                            cell_pos[dst, 1] = run_pos[p + local, 1]
                            cell_pos[dst, 2] = run_pos[p + local, 2]
                        hit_bufs = np.empty((n_threads, m), dtype=np.int32)
                        if direct_lo != -1:
                            repulsion_direct_range_3d(
                                run_pos, members, run_starts, run_ends, acc, repulsion, direct_lo, p,
                                k_coul, max_f, max_total_repulsion, cutoff_dist, cutoff_sq, taper_start, taper_width,
                            )
                            direct_lo = -1
                        repulsion_grid_run_3d(
                            run_pos, members, acc, repulsion, p, m,
                            cell_cx, cell_cy, cell_cz, nx, ny, nz, cell_ptr, cell_nodes, cell_pos, hit_bufs,
                            k_coul, max_f, max_total_repulsion, cutoff_dist, cutoff_sq, taper_start, taper_width,
                        )
                        used_grid = True
        if not used_grid and direct_lo == -1:
            direct_lo = p
        p = end

    if direct_lo != -1:
        repulsion_direct_range_3d(
            run_pos, members, run_starts, run_ends, acc, repulsion, direct_lo, n_active,
            k_coul, max_f, max_total_repulsion, cutoff_dist, cutoff_sq, taper_start, taper_width,
        )


@_compile_cached(fastmath=_UNFUSED_FASTMATH)
def euler_update_3d(pos, vel, acc, i, dt, damping):
    acc[i, 0] -= damping * vel[i, 0]; acc[i, 1] -= damping * vel[i, 1]; acc[i, 2] -= damping * vel[i, 2]
    vel[i, 0] += acc[i, 0] * dt; vel[i, 1] += acc[i, 1] * dt; vel[i, 2] += acc[i, 2] * dt
    pos[i, 0] += vel[i, 0] * dt; pos[i, 1] += vel[i, 1] * dt; pos[i, 2] += vel[i, 2] * dt


@_compile_cached
def run_physics_kernel_precomputed_3d(
    pos, vel, springs, active_mask, active_nodes, box_limits,
    members, run_starts, run_ends, acc, repulsion, run_pos, n_threads,
    dt, damping, k_spr, k_coul, max_f, max_total_repulsion, cutoff_dist,
):
    n_balls = pos.shape[0]
    for i in range(n_balls):
        acc[i, 0] = 0.0
        acc[i, 1] = 0.0
        acc[i, 2] = 0.0
        repulsion[i, 0] = 0.0
        repulsion[i, 1] = 0.0
        repulsion[i, 2] = 0.0

    for i in range(springs.shape[0]):
        idx_a, idx_b = springs[i, 0], springs[i, 1]
        if not active_mask[idx_a] or not active_mask[idx_b]:
            continue
        dx = pos[idx_a, 0] - pos[idx_b, 0]
        dy = pos[idx_a, 1] - pos[idx_b, 1]
        dz = pos[idx_a, 2] - pos[idx_b, 2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
        f = -k_spr * dist
        acc[idx_a, 0] += f * (dx / dist); acc[idx_a, 1] += f * (dy / dist); acc[idx_a, 2] += f * (dz / dist)
        acc[idx_b, 0] -= f * (dx / dist); acc[idx_b, 1] -= f * (dy / dist); acc[idx_b, 2] -= f * (dz / dist)

    accumulate_repulsion_grouped_3d(
        pos, acc, repulsion, run_pos, members, run_starts, run_ends, n_threads,
        k_coul, max_f, max_total_repulsion, cutoff_dist,
    )

    rmsd = 0.0
    n_active = active_nodes.shape[0]
    for i in range(n_balls):
        if not active_mask[i]:
            continue
        box_limit = box_limits[i]
        old_x, old_y, old_z = pos[i, 0], pos[i, 1], pos[i, 2]
        euler_update_3d(pos, vel, acc, i, dt, damping)

        if pos[i, 0] > box_limit: pos[i, 0] = box_limit; vel[i, 0] *= -0.5
        elif pos[i, 0] < -box_limit: pos[i, 0] = -box_limit; vel[i, 0] *= -0.5
        if pos[i, 1] > box_limit: pos[i, 1] = box_limit; vel[i, 1] *= -0.5
        elif pos[i, 1] < -box_limit: pos[i, 1] = -box_limit; vel[i, 1] *= -0.5
        if pos[i, 2] > box_limit: pos[i, 2] = box_limit; vel[i, 2] *= -0.5
        elif pos[i, 2] < -box_limit: pos[i, 2] = -box_limit; vel[i, 2] *= -0.5

        diff_x, diff_y, diff_z = pos[i, 0] - old_x, pos[i, 1] - old_y, pos[i, 2] - old_z
        rmsd += diff_x ** 2 + diff_y ** 2 + diff_z ** 2

    if n_active == 0:
        return 0.0
    return math.sqrt(rmsd / n_active)
