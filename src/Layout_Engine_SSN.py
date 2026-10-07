# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math
from collections import deque

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range

try:
    from utilities import Numba_Threads
except ImportError:
    import Numba_Threads

try:
    import torch
    try:
        from utilities import Hardware_Acceleration
        Hardware_Utils = Hardware_Acceleration
        Layout_Hardware = Hardware_Acceleration
    except ImportError:
        import Hardware_Acceleration as Hardware_Utils
        Layout_Hardware = Hardware_Utils
    HAS_TORCH = True
except Exception as e:
    import traceback
    print("Warning: PyTorch or Hardware_Utils could not be imported. GPU acceleration will be disabled.")
    print(f"Detail: {e}")
    traceback.print_exc()
    HAS_TORCH = False

# --- 1. Physics Kernels ---

def _compile_physics_kernel(kernel, parallel=False, fastmath=True):
    """JIT-compile a kernel when numba is present, else return the closure."""
    if NUMBA_AVAILABLE:
        return jit(nopython=True, fastmath=fastmath, parallel=parallel)(kernel)
    return kernel


# Every fastmath flag except 'contract'. The Euler update was once written as
# slice arithmetic (acc[i] -= damping * vel[i]), whose float64 temporaries
# rounded each product before it was added. Without 'contract' the per-axis
# update keeps those two roundings instead of fusing them into one FMA.
_UNFUSED_FASTMATH = {'nnan', 'ninf', 'nsz', 'arcp', 'afn', 'reassoc'}


def _group_by_component(comp_labels, active_nodes):
    """Group active nodes by component, keeping their order within each one.

    Returns the regrouped nodes and, for each position in that order, the
    bounds of its component's run. A node that walks its run in order meets
    the same partners in the same order as the historical all-pairs loop,
    which skipped every cross-component pair.
    """
    n_active = active_nodes.shape[0]
    order = np.argsort(comp_labels[active_nodes], kind='mergesort')
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


_group_active_nodes = _compile_physics_kernel(_group_by_component)


def _build_physics_kernel_2d():
    def _accumulate_repulsion(pos, acc, repulsion, comp_labels, active_nodes, k_coul, max_f, max_total_repulsion, cutoff_dist):
        # Calculate squared cutoff for efficient distance comparison
        cutoff_sq = cutoff_dist * cutoff_dist
        taper_start = cutoff_dist * 0.8
        taper_width = max(cutoff_dist * 0.2, 1e-9)

        # Each node sums its own pairs, so the threads never write the same
        # row. Summing in run order keeps every float rounding of the serial
        # pair loop, and the result does not depend on the thread count.
        members, run_starts, run_ends = _group_active_nodes(comp_labels, active_nodes)
        # A contiguous copy in run order keeps the inner loop off the
        # members[] indirection.
        run_pos = np.empty((members.shape[0], 2), dtype=pos.dtype)
        for position in range(members.shape[0]):
            run_pos[position, 0] = pos[members[position], 0]
            run_pos[position, 1] = pos[members[position], 1]

        for position in prange(members.shape[0]):
            node = members[position]
            for other_position in range(run_starts[position], run_ends[position]):
                if other_position == position:
                    continue

                dx = run_pos[position, 0] - run_pos[other_position, 0]
                dy = run_pos[position, 1] - run_pos[other_position, 1]
                dist_sq = dx*dx + dy*dy

                if dist_sq > cutoff_sq: continue
                if dist_sq == 0.0: continue

                dist = math.sqrt(dist_sq)
                safe_dist = max(dist, 0.5)

                f = k_coul / (safe_dist**2)

                if max_f > 0.0 and f > max_f:
                    f = max_f
                if dist > taper_start:
                    f *= max(0.0, (cutoff_dist - dist) / taper_width)

                repulsion[node, 0] += f*(dx/dist); repulsion[node, 1] += f*(dy/dist)

            # MAX_FORCE_LIMIT caps each pair. This second cap limits the norm of the
            # accumulated repulsive force on a node before it is combined with springs.
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

    # Only the O(N^2) repulsion runs in parallel. Springs scatter into shared
    # rows and are O(E), so they stay serial with the O(N) integration.
    accumulate_repulsion = _compile_physics_kernel(
        _accumulate_repulsion, parallel=True
    )

    def _euler_update(pos, vel, acc, i, dt, damping):
        acc[i, 0] -= damping * vel[i, 0]; acc[i, 1] -= damping * vel[i, 1]
        vel[i, 0] += acc[i, 0] * dt; vel[i, 1] += acc[i, 1] * dt
        pos[i, 0] += vel[i, 0] * dt; pos[i, 1] += vel[i, 1] * dt

    # Per-axis scalars avoid allocating slice temporaries for every node on
    # every step, which used to dominate batches of small components.
    euler_update = _compile_physics_kernel(
        _euler_update, fastmath=_UNFUSED_FASTMATH
    )

    def _run_physics_kernel(pos, vel, springs, comp_labels, active_mask, active_nodes, box_limits, dt, damping, k_spr, k_coul, max_f, max_total_repulsion, cutoff_dist):
        n_balls = pos.shape[0]
        acc = np.zeros_like(pos)
        repulsion = np.zeros_like(pos)

        # --- SPRINGS (Attraction) ---
        for i in range(springs.shape[0]):
            idx_a, idx_b = springs[i, 0], springs[i, 1]
            if not active_mask[idx_a] or not active_mask[idx_b]:
                continue
            dx, dy = pos[idx_a, 0] - pos[idx_b, 0], pos[idx_a, 1] - pos[idx_b, 1]
            dist = math.sqrt(dx*dx + dy*dy) + 1e-9
            
            f = -k_spr * dist
            
            acc[idx_a, 0] += f * (dx/dist); acc[idx_a, 1] += f * (dy/dist)
            acc[idx_b, 0] -= f * (dx/dist); acc[idx_b, 1] -= f * (dy/dist)

        # --- REPULSION (Coulomb Only) ---
        accumulate_repulsion(
            pos, acc, repulsion, comp_labels, active_nodes,
            k_coul, max_f, max_total_repulsion, cutoff_dist,
        )

        # --- INTEGRATION (Euler) ---
        rmsd = 0.0
        n_active = active_nodes.shape[0]
        for i in range(n_balls):
            if not active_mask[i]:
                continue
            box_limit = box_limits[i]
            old_x, old_y = pos[i, 0], pos[i, 1]
            euler_update(pos, vel, acc, i, dt, damping)

            if pos[i,0] > box_limit: pos[i,0]=box_limit; vel[i,0]*=-0.5
            elif pos[i,0] < -box_limit: pos[i,0]=-box_limit; vel[i,0]*=-0.5
            if pos[i,1] > box_limit: pos[i,1]=box_limit; vel[i,1]*=-0.5
            elif pos[i,1] < -box_limit: pos[i,1]=-box_limit; vel[i,1]*=-0.5

            diff_x, diff_y = pos[i, 0] - old_x, pos[i, 1] - old_y
            rmsd += diff_x**2 + diff_y**2

        if n_active == 0:
            return 0.0
        return math.sqrt(rmsd / n_active)
        
    return _compile_physics_kernel(_run_physics_kernel)


def _build_physics_kernel_3d():
    """Line-for-line 3D twin of the 2D kernel.

    The axis components are written out explicitly rather than looped over a
    runtime-length dimension so numba keeps the same fully unrolled codegen it
    produces for the 2D path. Every formula, clamp and ordering below matches
    _build_physics_kernel_2d exactly; only the z terms are added.
    """
    def _accumulate_repulsion(pos, acc, repulsion, comp_labels, active_nodes, k_coul, max_f, max_total_repulsion, cutoff_dist):
        # Calculate squared cutoff for efficient distance comparison
        cutoff_sq = cutoff_dist * cutoff_dist
        taper_start = cutoff_dist * 0.8
        taper_width = max(cutoff_dist * 0.2, 1e-9)

        members, run_starts, run_ends = _group_active_nodes(comp_labels, active_nodes)
        run_pos = np.empty((members.shape[0], 3), dtype=pos.dtype)
        for position in range(members.shape[0]):
            run_pos[position, 0] = pos[members[position], 0]
            run_pos[position, 1] = pos[members[position], 1]
            run_pos[position, 2] = pos[members[position], 2]

        for position in prange(members.shape[0]):
            node = members[position]
            for other_position in range(run_starts[position], run_ends[position]):
                if other_position == position:
                    continue

                dx = run_pos[position, 0] - run_pos[other_position, 0]
                dy = run_pos[position, 1] - run_pos[other_position, 1]
                dz = run_pos[position, 2] - run_pos[other_position, 2]
                dist_sq = dx*dx + dy*dy + dz*dz

                if dist_sq > cutoff_sq: continue
                if dist_sq == 0.0: continue

                dist = math.sqrt(dist_sq)
                safe_dist = max(dist, 0.5)

                f = k_coul / (safe_dist**2)

                if max_f > 0.0 and f > max_f:
                    f = max_f
                if dist > taper_start:
                    f *= max(0.0, (cutoff_dist - dist) / taper_width)

                repulsion[node, 0] += f*(dx/dist); repulsion[node, 1] += f*(dy/dist); repulsion[node, 2] += f*(dz/dist)

            # MAX_FORCE_LIMIT caps each pair. This second cap limits the norm of the
            # accumulated repulsive force on a node before it is combined with springs.
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

    accumulate_repulsion = _compile_physics_kernel(
        _accumulate_repulsion, parallel=True
    )

    def _euler_update(pos, vel, acc, i, dt, damping):
        acc[i, 0] -= damping * vel[i, 0]; acc[i, 1] -= damping * vel[i, 1]; acc[i, 2] -= damping * vel[i, 2]
        vel[i, 0] += acc[i, 0] * dt; vel[i, 1] += acc[i, 1] * dt; vel[i, 2] += acc[i, 2] * dt
        pos[i, 0] += vel[i, 0] * dt; pos[i, 1] += vel[i, 1] * dt; pos[i, 2] += vel[i, 2] * dt

    euler_update = _compile_physics_kernel(
        _euler_update, fastmath=_UNFUSED_FASTMATH
    )

    def _run_physics_kernel(pos, vel, springs, comp_labels, active_mask, active_nodes, box_limits, dt, damping, k_spr, k_coul, max_f, max_total_repulsion, cutoff_dist):
        n_balls = pos.shape[0]
        acc = np.zeros_like(pos)
        repulsion = np.zeros_like(pos)

        # --- SPRINGS (Attraction) ---
        for i in range(springs.shape[0]):
            idx_a, idx_b = springs[i, 0], springs[i, 1]
            if not active_mask[idx_a] or not active_mask[idx_b]:
                continue
            dx = pos[idx_a, 0] - pos[idx_b, 0]
            dy = pos[idx_a, 1] - pos[idx_b, 1]
            dz = pos[idx_a, 2] - pos[idx_b, 2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9

            f = -k_spr * dist

            acc[idx_a, 0] += f * (dx/dist); acc[idx_a, 1] += f * (dy/dist); acc[idx_a, 2] += f * (dz/dist)
            acc[idx_b, 0] -= f * (dx/dist); acc[idx_b, 1] -= f * (dy/dist); acc[idx_b, 2] -= f * (dz/dist)

        # --- REPULSION (Coulomb Only) ---
        accumulate_repulsion(
            pos, acc, repulsion, comp_labels, active_nodes,
            k_coul, max_f, max_total_repulsion, cutoff_dist,
        )

        # --- INTEGRATION (Euler) ---
        rmsd = 0.0
        n_active = active_nodes.shape[0]
        for i in range(n_balls):
            if not active_mask[i]:
                continue
            box_limit = box_limits[i]
            old_x, old_y, old_z = pos[i, 0], pos[i, 1], pos[i, 2]
            euler_update(pos, vel, acc, i, dt, damping)

            if pos[i,0] > box_limit: pos[i,0]=box_limit; vel[i,0]*=-0.5
            elif pos[i,0] < -box_limit: pos[i,0]=-box_limit; vel[i,0]*=-0.5
            if pos[i,1] > box_limit: pos[i,1]=box_limit; vel[i,1]*=-0.5
            elif pos[i,1] < -box_limit: pos[i,1]=-box_limit; vel[i,1]*=-0.5
            if pos[i,2] > box_limit: pos[i,2]=box_limit; vel[i,2]*=-0.5
            elif pos[i,2] < -box_limit: pos[i,2]=-box_limit; vel[i,2]*=-0.5

            diff_x, diff_y, diff_z = pos[i, 0] - old_x, pos[i, 1] - old_y, pos[i, 2] - old_z
            rmsd += diff_x**2 + diff_y**2 + diff_z**2

        if n_active == 0:
            return 0.0
        return math.sqrt(rmsd / n_active)

    return _compile_physics_kernel(_run_physics_kernel)


_PHYSICS_KERNEL_BUILDERS = {
    2: _build_physics_kernel_2d,
    3: _build_physics_kernel_3d,
}
_PHYSICS_KERNEL_CACHE = {}


def _get_physics_kernel(dimensions=2):
    """Return the physics kernel specialized for `dimensions` coordinates."""
    key = int(dimensions)
    builder = _PHYSICS_KERNEL_BUILDERS.get(key)
    if builder is None:
        raise ValueError(
            f"Unsupported layout dimensionality {dimensions!r}; expected 2 or 3."
        )
    kernel = _PHYSICS_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = builder()
        _PHYSICS_KERNEL_CACHE[key] = kernel
    return kernel


# Backward-compatible alias: the 2D kernel remains importable under its
# historical name for callers that predate the dimensionality switch.
run_physics_kernel = _get_physics_kernel(2)

def _normalize_active_mask(active_mask, n_nodes):
    if active_mask is None:
        return np.ones(n_nodes, dtype=np.bool_)

    mask = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
    if mask.size != n_nodes:
        raise ValueError("active_mask must contain one value per node.")
    return mask


def _normalize_box_limits(box_limits, n_nodes):
    """Expand a scalar boundary or validate a per-node boundary array."""
    limits = np.asarray(box_limits, dtype=np.float32)
    if limits.ndim == 0:
        return np.full(n_nodes, float(limits), dtype=np.float32)

    limits = limits.reshape(-1)
    if limits.size != n_nodes:
        raise ValueError("box_limits must be a scalar or contain one value per node.")
    return limits


class SSNSimulationCPU:
    def __init__(self, pos, springs, comp_labels, box_limit, params, active_mask=None):
        self.pos = pos.astype(np.float32)
        self.vel = np.zeros_like(pos)
        self.springs = springs
        self.comp_labels = np.asarray(comp_labels, dtype=np.int32)
        self.active_mask = _normalize_active_mask(active_mask, len(self.pos))
        self.active_nodes = np.flatnonzero(self.active_mask).astype(np.int32)
        self.box_limits = _normalize_box_limits(box_limit, len(self.pos))
        self.params = params
        self.cutoff = float(self.params.get('COULOMB_CUTOFF', 15.0))
        # Dimensionality is carried by the position array itself, so the
        # simulation-object protocol is unchanged for 2D callers.
        self.dimensions = int(self.pos.shape[1])
        self._kernel = _get_physics_kernel(self.dimensions)
        self.threads = Numba_Threads.default_thread_count()

    def step(self, current_step):
        with Numba_Threads.limited_threads(self.threads):
            return self._kernel(
                self.pos, self.vel, self.springs, self.comp_labels,
                self.active_mask, self.active_nodes, self.box_limits,
                self.params.get('DT', 0.1),
                self.params.get('DAMPING', 0.5),
                self.params.get('SPRING_K', 0.1),
                self.params.get('COULOMB_K', 50.0),
                self.params.get('MAX_FORCE_LIMIT', 20.0),
                self.params.get('MAX_TOTAL_REPULSION_FORCE', 0.0),
                self.cutoff
            )
        
    def get_pos(self): return self.pos

if HAS_TORCH:
    class SSNSimulationGPU:
        def __init__(self, pos, springs, comp_labels, box_limit, params, active_mask=None, *, device=None):
            # Production callers always pass an explicit candidate. CPU is a
            # safe compatibility default for direct/unit-test construction.
            self.device = torch.device("cpu") if device is None else device
            self.pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
            self.vel = torch.zeros_like(self.pos)
            self.springs = torch.tensor(springs, dtype=torch.long, device=self.device)
            self.comp_labels = torch.tensor(comp_labels, dtype=torch.long, device=self.device)
            active_mask_array = _normalize_active_mask(active_mask, len(pos))
            self.active_mask = torch.tensor(
                active_mask_array,
                dtype=torch.bool,
                device=self.device
            )
            self.box_limits = torch.tensor(
                _normalize_box_limits(box_limit, len(pos)),
                dtype=torch.float32,
                device=self.device
            )
            self.params = params
            self.cutoff = float(self.params.get('COULOMB_CUTOFF', 15.0))
        
        @torch.no_grad()
        def step(self, current_step):
            # --- PHYSICS ---
            delta = self.pos.unsqueeze(1) - self.pos.unsqueeze(0)
            dist_sq = (delta * delta).sum(dim=2)
            dist = torch.sqrt(dist_sq)
            direction_denominator = torch.where(
                dist_sq > 0.0,
                dist,
                torch.ones_like(dist),
            )

            pair_mask = (
                self.active_mask.unsqueeze(1)
                & self.active_mask.unsqueeze(0)
                & (self.comp_labels.unsqueeze(1) == self.comp_labels.unsqueeze(0))
                & (dist_sq > 0.0)
                & (dist_sq <= self.cutoff * self.cutoff)
            )

            f_mag = (
                self.params.get('COULOMB_K', 50.0)
                / dist.clamp(min=0.5).pow(2)
            )
            max_f = self.params.get('MAX_FORCE_LIMIT', 20.0)
            if max_f > 0.0:
                f_mag.clamp_(max=max_f)
            taper_start = self.cutoff * 0.8
            taper_width = max(self.cutoff * 0.2, 1e-9)
            taper = (
                (self.cutoff - dist) / taper_width
            ).clamp(min=0.0, max=1.0)
            taper = torch.where(dist > taper_start, taper, 1.0)
            f_mag = torch.where(pair_mask, f_mag * taper, 0.0)

            repulsion = (
                f_mag.unsqueeze(2)
                * (delta / direction_denominator.unsqueeze(2))
            ).sum(dim=1)

            max_total_repulsion = self.params.get('MAX_TOTAL_REPULSION_FORCE', 0.0)
            if max_total_repulsion > 0.0:
                repulsion_norm = repulsion.norm(dim=1, keepdim=True)
                repulsion_scale = (
                    max_total_repulsion / repulsion_norm.clamp(min=1e-12)
                ).clamp(max=1.0)
                repulsion *= repulsion_scale
            acc = repulsion
            
            if len(self.springs) > 0:
                idx_a, idx_b = self.springs[:,0], self.springs[:,1]
                spring_active = self.active_mask[idx_a] & self.active_mask[idx_b]
                idx_a = idx_a[spring_active]
                idx_b = idx_b[spring_active]
                pa, pb = self.pos[idx_a], self.pos[idx_b]
                d = (pa-pb).norm(dim=1) + 1e-9
                f = -self.params.get('SPRING_K', 0.1) * d
                fv = f.unsqueeze(1) * ((pa-pb)/d.unsqueeze(1))
                acc.index_add_(0, idx_a, fv); acc.index_add_(0, idx_b, -fv)
            
            damping = self.params.get('DAMPING', 0.5)
            dt = self.params.get('DT', 0.1)
            
            acc -= damping * self.vel
            acc[~self.active_mask] = 0.0
            self.vel[~self.active_mask] = 0.0
            self.vel[self.active_mask] += acc[self.active_mask] * dt
            old = self.pos.clone()
            self.pos[self.active_mask] += self.vel[self.active_mask] * dt
            
            # --- Boundary Collisions (Match CPU Bouncing) ---
            # Reverse and dampen velocity for nodes hitting the walls.
            # Every other tensor op in this class is already rank-agnostic, so
            # iterating the axes is all that 3D support requires here.
            for axis in range(self.pos.shape[1]):
                out_of_bounds = (
                    self.pos[:, axis].abs() > self.box_limits
                ) & self.active_mask
                self.vel[out_of_bounds, axis] *= -0.5
            
            # Clamp positions
            limits = self.box_limits[self.active_mask].unsqueeze(1)
            active_pos = self.pos[self.active_mask]
            self.pos[self.active_mask] = torch.maximum(
                torch.minimum(active_pos, limits),
                -limits
            )

            if self.active_mask.any():
                rmsd = (
                    (self.pos[self.active_mask] - old[self.active_mask])
                    .norm(dim=1).pow(2).mean().sqrt().item()
                )
            else:
                rmsd = 0.0
            
            del delta, dist_sq, dist, direction_denominator
            del pair_mask, f_mag, taper
            del repulsion, acc, old, limits, active_pos
            if max_total_repulsion > 0.0:
                del repulsion_norm, repulsion_scale
            return rmsd

        def get_pos(self): return self.pos.cpu().numpy()

# --- 2. Components & Packing Logic ---

def _compile_serial_helper(function):
    """JIT-compile a serial helper when numba is present, else keep it in Python."""
    if NUMBA_AVAILABLE:
        return jit(nopython=True, cache=True)(function)
    return function


def _edge_array(edges, n_nodes):
    """Return edges as an (m, 2) integer array of valid node indices."""
    edges = np.asarray(edges)
    if edges.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if not np.issubdtype(edges.dtype, np.integer):
        edges = edges.astype(np.int64)
    edges = edges.reshape(-1, 2)
    if edges.min() < 0 or edges.max() >= n_nodes:
        raise IndexError("Edge endpoints must be node indices below n_nodes.")
    return edges


@_compile_serial_helper
def _adjacency_lists(n_nodes, sources, targets):
    """CSR neighbour lists, each in the order the edge list reaches the node.

    Edge (u, v) appends v to u's list, then u to v's, as the original
    dict-of-lists adjacency did.
    """
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    for edge in range(sources.shape[0]):
        indptr[sources[edge] + 1] += 1
        indptr[targets[edge] + 1] += 1
    for node in range(n_nodes):
        indptr[node + 1] += indptr[node]
    fill = indptr[:-1].copy()
    neighbours = np.empty(indptr[n_nodes], dtype=np.int32)
    for edge in range(sources.shape[0]):
        u = sources[edge]
        v = targets[edge]
        neighbours[fill[u]] = v
        fill[u] += 1
        neighbours[fill[v]] = u
        fill[v] += 1
    return indptr, neighbours


@_compile_serial_helper
def _breadth_first_order(n_nodes, indptr, neighbours):
    """Visit every component breadth-first from its smallest node.

    Returns the visit order and the offset at which each component starts.
    """
    visited = np.zeros(n_nodes, dtype=np.bool_)
    order = np.empty(n_nodes, dtype=np.int64)
    starts = np.empty(n_nodes + 1, dtype=np.int64)
    count = 0
    tail = 0
    for root in range(n_nodes):
        if visited[root]:
            continue
        starts[count] = tail
        count += 1
        visited[root] = True
        order[tail] = root
        tail += 1
        head = starts[count - 1]
        while head < tail:
            node = order[head]
            head += 1
            for slot in range(indptr[node], indptr[node + 1]):
                neighbour = neighbours[slot]
                if not visited[neighbour]:
                    visited[neighbour] = True
                    order[tail] = neighbour
                    tail += 1
    starts[count] = tail
    return order, starts[:count + 1]


@_compile_serial_helper
def _stable_group_order(labels, n_groups):
    """Counting sort of positions by label; negative labels are dropped.

    The positions labelled g are order[bounds[g]:bounds[g + 1]], in their
    original order.
    """
    bounds = np.zeros(n_groups + 1, dtype=np.int64)
    for position in range(labels.shape[0]):
        label = labels[position]
        if label >= 0:
            bounds[label + 1] += 1
    for group in range(n_groups):
        bounds[group + 1] += bounds[group]
    fill = bounds[:-1].copy()
    order = np.empty(bounds[n_groups], dtype=np.int64)
    for position in range(labels.shape[0]):
        label = labels[position]
        if label >= 0:
            order[fill[label]] = position
            fill[label] += 1
    return order, bounds


def find_connected_components(n_nodes, edges):
    """Return every connected component as an array of node indices.

    Components come in order of their smallest node, and the nodes of each in
    breadth-first order from it, taking neighbours in edge-list order. That is
    the order the original per-edge Python search produced, and seeded layouts
    depend on it: a batch lays its nodes out in this order.
    """
    n_nodes = int(n_nodes)
    if n_nodes <= 0:
        return []
    edges = _edge_array(edges, n_nodes)
    indptr, neighbours = _adjacency_lists(n_nodes, edges[:, 0], edges[:, 1])
    order, starts = _breadth_first_order(n_nodes, indptr, neighbours)
    return [order[starts[k]:starts[k + 1]] for k in range(len(starts) - 1)]

def get_component_labels(n_nodes, edges):
    """Maps each node to its connected component ID for isolated physics."""
    labels = np.zeros(n_nodes, dtype=np.int32)
    for c_id, comp in enumerate(find_connected_components(n_nodes, edges)):
        labels[comp] = c_id
    return labels

def _resolve_seed(params):
    """Return the layout seed, or None for non-reproducible behaviour."""
    value = params.get('LAYOUT_SEED', 42)
    if value is None:
        return None
    seed = int(value)
    if seed < 0:
        raise ValueError(
            f"LAYOUT_SEED must be a non-negative integer or null, got {value!r}."
        )
    return seed


def _job_generator(seed, job_index):
    """Independent stream per job.

    Spawning per job rather than sharing one Generator means adding or
    removing a component cannot shift the random draws of every component
    scheduled after it.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng([seed, job_index])


def _resolve_dimensions(params):
    """Read the coordinate count for this run. Defaults to the historical 2D."""
    value = int(params.get('LAYOUT_DIMENSIONS', 2) or 2)
    if value not in (2, 3):
        raise ValueError(
            f"LAYOUT_DIMENSIONS must be 2 or 3, got {value!r}."
        )
    return value


def _initial_lattice(n_nodes, box_limit, dimensions):
    """Uniform lattice seed. For dimensions=2 this reproduces the 2D meshgrid."""
    if n_nodes <= 0:
        return np.zeros((0, dimensions), dtype=np.float32)
    side = max(int(np.ceil(n_nodes ** (1.0 / dimensions))), 1)
    axis = np.linspace(-box_limit * 0.5, box_limit * 0.5, side)
    mesh = np.meshgrid(*([axis] * dimensions))
    lattice = np.column_stack([component.flatten() for component in mesh])
    if lattice.shape[0] < n_nodes:
        repeats = int(np.ceil(n_nodes / lattice.shape[0]))
        lattice = np.tile(lattice, (repeats, 1))
    return lattice[:n_nodes].astype(np.float32)


def _fibonacci_sphere(count):
    """Near-uniform unit vectors on the sphere via the golden-angle spiral."""
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if count == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    index = np.arange(count, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * index / (count - 1)
    ring = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    theta = golden_angle * index
    return np.column_stack((np.cos(theta) * ring, np.sin(theta) * ring, z))


def pack_components_to_shells(pos, edges, n_nodes, spacing, padding):
    """Distribute independent components over concentric spherical shells.

    The 2D packer tiles components into a readable poster because a flat view
    has to show everything at once. In a headset the viewer flies between
    components, so the 3D analogue optimizes for navigability instead of area:
    the largest component sits at the origin and the rest are distributed over
    shells around it, each placed on a golden-angle direction that clears every
    component already placed.
    """
    print("Packing independent components onto concentric spherical shells...")
    components = find_connected_components(n_nodes, edges)

    if not components:
        return pos, 100.0

    gap = max(float(spacing), 0.0)
    pad = max(float(padding), 0.0) / 2.0

    component_info = []
    for component in components:
        indices = np.asarray(component, dtype=np.int64)
        block = pos[indices].astype(np.float64)
        centroid = block.mean(axis=0)
        local = block - centroid
        extent = (
            float(np.max(np.linalg.norm(local, axis=1))) if indices.size else 0.0
        )
        component_info.append({
            'indices': indices,
            'local': local,
            'radius': max(extent, 1e-6) + pad,
            'num_nodes': int(indices.size),
        })

    # Largest first, matching the 2D packer's ordering intent.
    component_info.sort(key=lambda c: (c['radius'], c['num_nodes']), reverse=True)

    new_pos = np.asarray(pos, dtype=np.float64).copy()
    placed_centers = np.zeros((len(component_info), 3), dtype=np.float64)
    placed_radii = np.zeros(len(component_info), dtype=np.float64)
    placed_count = 0

    def _place(component, center):
        nonlocal placed_count
        new_pos[component['indices']] = component['local'] + center
        placed_centers[placed_count] = center
        placed_radii[placed_count] = component['radius']
        placed_count += 1

    _place(component_info[0], np.zeros(3, dtype=np.float64))

    pending = component_info[1:]
    cursor = 0
    shell = 0
    shell_radius = component_info[0]['radius']

    while cursor < len(pending):
        shell += 1
        shell_radius += max(c['radius'] for c in pending[cursor:]) + gap
        # Candidate density grows with the shell's surface area.
        directions = _fibonacci_sphere(max(16, 10 * shell * shell))
        for direction in directions:
            if cursor >= len(pending):
                break
            component = pending[cursor]
            center = direction * shell_radius
            required = placed_radii[:placed_count] + component['radius'] + gap
            offsets = placed_centers[:placed_count] - center
            separation = np.einsum('ij,ij->i', offsets, offsets)
            if np.all(separation >= required * required):
                _place(component, center)
                cursor += 1

    global_min = np.min(new_pos, axis=0)
    global_max = np.max(new_pos, axis=0)
    new_pos -= (global_max + global_min) / 2.0

    new_box_limit = float(np.max(global_max - global_min)) / 2.0 * 1.1

    print(
        f"Packed {len(components)} objects onto {shell} shells. "
        "Ready for display."
    )
    return new_pos.astype(np.float32), new_box_limit


@_compile_serial_helper
def _first_cell(x):
    """Index of the raster cell holding x, clamped to the first cell."""
    return int(x) if x > 0.0 else 0


@_compile_serial_helper
def _last_cell(x, first):
    """Last cell reached by an interval that ends just before x, never before `first`.

    Leaving x itself out keeps a reach that stops exactly on a cell border
    from claiming the next cell.
    """
    last = int(math.ceil(x)) - 1
    return last if last > first else first


@_compile_serial_helper
def _mark_footprint(u, v, edges, reach, mask):
    """Mark the cells within `reach` of every node and edge of one component.

    Coordinates are in cell units from the component's top-left corner, u to
    the right and v downward, and `reach` is a half-width in the same units.
    A node claims the cells under its reach box. An edge claims, row by row,
    the cells within reach of the stretch of the segment near that row, so
    the cost follows the cells an edge covers rather than its drawn length.
    """
    rows = mask.shape[0]
    cols = mask.shape[1]
    for node in range(u.shape[0]):
        c0 = _first_cell(u[node] - reach)
        c1 = min(_last_cell(u[node] + reach, c0), cols - 1)
        r0 = _first_cell(v[node] - reach)
        r1 = min(_last_cell(v[node] + reach, r0), rows - 1)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                mask[r, c] = True
    for edge in range(edges.shape[0]):
        a = edges[edge, 0]
        b = edges[edge, 1]
        u1 = u[a]
        v1 = v[a]
        du = u[b] - u1
        dv = v[b] - v1
        r0 = _first_cell(min(v1, v[b]) - reach)
        r1 = min(_last_cell(max(v1, v[b]) + reach, r0), rows - 1)
        for r in range(r0, r1 + 1):
            t0 = 0.0
            t1 = 1.0
            if dv != 0.0:
                # The part of the segment whose reach overlaps row r.
                t0 = (r - reach - v1) / dv
                t1 = (r + 1.0 + reach - v1) / dv
                if t0 > t1:
                    t0, t1 = t1, t0
                t0 = max(t0, 0.0)
                t1 = min(t1, 1.0)
                if t0 > t1:
                    continue
            ua = u1 + t0 * du
            ub = u1 + t1 * du
            if ua > ub:
                ua, ub = ub, ua
            c0 = _first_cell(ua - reach)
            c1 = min(_last_cell(ub + reach, c0), cols - 1)
            for c in range(c0, c1 + 1):
                mask[r, c] = True


def _component_footprint(shifted_pos, local_edges, cell, reach):
    """Boolean raster of the cells one component claims (see _mark_footprint).

    shifted_pos has its top-left corner at the origin (x >= 0, y <= 0).
    """
    u = shifted_pos[:, 0].astype(np.float64) / cell
    v = -shifted_pos[:, 1].astype(np.float64) / cell
    u_max = float(u.max())
    v_max = float(v.max())
    cols = _last_cell(u_max + reach, _first_cell(u_max)) + 1
    rows = _last_cell(v_max + reach, _first_cell(v_max)) + 1
    mask = np.zeros((rows, cols), dtype=np.bool_)
    _mark_footprint(u, v, local_edges, reach, mask)
    return mask


@_compile_serial_helper
def _first_fit(grid, spiral_rows, spiral_cols, start, cell_rows, cell_cols, height, width):
    """Place a footprint at the first spiral position, from `start`, where it fits.

    Each spiral entry is a target cell for the footprint's centre. Returns
    the spiral index used and the top-left corner, or -1 when nothing fits.
    The cell that blocked the previous position is tested first, because it
    usually blocks the next one too.
    """
    size = grid.shape[0]
    blocker = 0
    for index in range(start, spiral_rows.shape[0]):
        top = spiral_rows[index] - height // 2
        left = spiral_cols[index] - width // 2
        if top < 0 or left < 0 or top + height > size or left + width > size:
            continue
        if grid[top + cell_rows[blocker], left + cell_cols[blocker]]:
            continue
        fits = True
        for cell in range(cell_rows.shape[0]):
            if grid[top + cell_rows[cell], left + cell_cols[cell]]:
                blocker = cell
                fits = False
                break
        if fits:
            for cell in range(cell_rows.shape[0]):
                grid[top + cell_rows[cell], left + cell_cols[cell]] = True
            return index, top, left
    return -1, 0, 0


def _spiral_order(size, round_envelope):
    """Every grid cell, ordered by distance from the centre.

    Square packing orders by Chebyshev distance (ties by Euclidean), so the
    packed area grows as a square; Circle orders by Euclidean distance, so it
    grows as a disc. Both use the same square cells as the footprints.
    """
    rows, cols = np.divmod(np.arange(size * size, dtype=np.int64), size)
    centre = size // 2
    d_rows = rows - centre
    d_cols = cols - centre
    euclidean = d_rows * d_rows + d_cols * d_cols
    if round_envelope:
        order = np.argsort(euclidean, kind='stable')
    else:
        chebyshev = np.maximum(np.abs(d_rows), np.abs(d_cols))
        order = np.lexsort((euclidean, chebyshev))
    return rows[order], cols[order]


def _place_footprints(footprints, size, round_envelope):
    """First-fit every footprint on a size x size grid; None if one does not fit."""
    grid = np.zeros((size, size), dtype=np.bool_)
    spiral_rows, spiral_cols = _spiral_order(size, round_envelope)
    # The grid only fills up, so every position before an identical
    # footprint's last placement is still blocked: its next copy resumes there.
    resume = {}
    corners = []
    for footprint in footprints:
        key = footprint['shape_key']
        index, top, left = _first_fit(
            grid, spiral_rows, spiral_cols, resume.get(key, 0),
            footprint['cell_rows'], footprint['cell_cols'],
            footprint['rows'], footprint['cols'],
        )
        if index < 0:
            return None
        resume[key] = index + 1
        corners.append((top, left))
    return corners


def pack_components_to_grid(pos, edges, n_nodes, grid_size, padding, packing_geometry="Square"):
    """Pack independent components onto a shared raster, largest first.

    A component claims every cell, `grid_size` units wide, within padding / 2
    of its nodes and edges, so components interlock without their drawings
    touching. Each is placed at the first position along a spiral from the
    centre where its cells are free: a square spiral for "Square", a round one
    for "Circle". Smaller cells pack more tightly.
    """
    print("Packing independent components onto the packing grid...")
    components = find_connected_components(n_nodes, edges)

    if not components:
        return pos, 100.0
    if not np.isfinite(pos).all():
        raise ValueError("Layout positions must be finite before packing.")

    cell = float(grid_size)
    reach = float(padding) / 2.0 / cell
    edges = _edge_array(edges, n_nodes)
    labels = np.empty(n_nodes, dtype=np.int64)
    local_index = np.empty(n_nodes, dtype=np.int64)
    for c_id, comp in enumerate(components):
        labels[comp] = c_id
        local_index[comp] = np.arange(len(comp))
    edge_order, edge_bounds = _stable_group_order(labels[edges[:, 0]], len(components))

    footprints = []
    for c_id, idx in enumerate(components):
        comp_pos = pos[idx]
        # Use a top-left origin so that x runs along columns and -y down rows.
        min_x = np.min(comp_pos[:, 0])
        max_y = np.max(comp_pos[:, 1])
        shifted_pos = comp_pos - [min_x, max_y]
        members = edge_order[edge_bounds[c_id]:edge_bounds[c_id + 1]]
        mask = _component_footprint(
            shifted_pos, local_index[edges[members]], cell, reach
        )
        cell_rows, cell_cols = np.nonzero(mask)
        footprints.append({
            'indices': idx,
            'shifted_pos': shifted_pos,
            'rows': mask.shape[0],
            'cols': mask.shape[1],
            'cell_rows': cell_rows,
            'cell_cols': cell_cols,
            'shape_key': (mask.shape, mask.tobytes()),
            'area': len(cell_rows),
            'num_nodes': len(idx),
        })
    del edge_order, labels, local_index

    # Largest area first, then most nodes; ties keep component order.
    footprints.sort(key=lambda footprint: (footprint['area'], footprint['num_nodes']), reverse=True)

    round_envelope = str(packing_geometry).lower() == "circle"
    total_area = sum(footprint['area'] for footprint in footprints)
    size = max(
        int(math.ceil(math.sqrt(total_area) * (2.0 if round_envelope else 1.5))),
        max(footprint['cols'] for footprint in footprints),
        max(footprint['rows'] for footprint in footprints),
    )
    corners = _place_footprints(footprints, size, round_envelope)
    while corners is None:
        size = int(size * 1.1) + 2
        corners = _place_footprints(footprints, size, round_envelope)

    new_pos = np.zeros((n_nodes, 2), dtype=np.float32)
    new_pos[:] = pos[:]
    for footprint, (top, left) in zip(footprints, corners):
        new_pos[footprint['indices'], 0] = footprint['shifted_pos'][:, 0] + left * cell
        new_pos[footprint['indices'], 1] = footprint['shifted_pos'][:, 1] - top * cell

    # --- Center the final visualization ---
    global_min = np.min(new_pos, axis=0)
    global_max = np.max(new_pos, axis=0)
    center = (global_max + global_min) / 2.0
    new_pos -= center

    new_box_limit = max(global_max[0] - global_min[0], global_max[1] - global_min[1]) / 2.0 * 1.1

    print(
        f"Packed {len(components)} components on a {size} x {size} grid "
        f"of {cell:g}-unit cells. Ready for display."
    )
    return new_pos, new_box_limit

# --- 3. Main Layout Algorithm ---

@_compile_serial_helper
def _accumulate_stage_anchors(
    edges, weights, scaled_weights, newly_active, previous_active,
    reference_pos, weighted_sum, weight_sum,
):
    """Sum each newly active node's already relaxed neighbours, in edge order.

    Matches the original per-edge NumPy loop bit for bit: a neighbour's
    position is scaled in the position dtype (NumPy converts the float weight
    to it first), then added in float64.
    """
    dimensions = reference_pos.shape[1]
    for edge in range(edges.shape[0]):
        u = edges[edge, 0]
        v = edges[edge, 1]
        if newly_active[u] and previous_active[v]:
            for axis in range(dimensions):
                weighted_sum[u, axis] += reference_pos[v, axis] * scaled_weights[edge]
            weight_sum[u] += weights[edge]
        if newly_active[v] and previous_active[u]:
            for axis in range(dimensions):
                weighted_sum[v, axis] += reference_pos[u, axis] * scaled_weights[edge]
            weight_sum[v] += weights[edge]


def _prepare_progressive_stage(
    pos, stage_edges, stage_scores, previous_active, rng=None
):
    """Activate stage nodes and place newly introduced nodes near active neighbors."""
    stage_edges = _edge_array(stage_edges, len(pos))
    previous_active = np.asarray(previous_active, dtype=np.bool_)
    active_mask = np.zeros(len(pos), dtype=np.bool_)
    active_mask[stage_edges[:, 0]] = True
    active_mask[stage_edges[:, 1]] = True

    newly_active = active_mask & (~previous_active)
    if not np.any(newly_active) or not np.any(previous_active):
        return active_mask

    reference_pos = pos.copy()
    weighted_sum = np.zeros_like(pos, dtype=np.float64)
    weight_sum = np.zeros(len(pos), dtype=np.float64)

    # Prefer neighbors that were already relaxed in the preceding stage.
    weights = np.maximum(
        np.asarray(stage_scores, dtype=np.float64).reshape(-1), 1e-9
    )
    if len(weights) != len(stage_edges):
        raise ValueError("stage_scores must hold one score per stage edge.")
    _accumulate_stage_anchors(
        stage_edges, weights, weights.astype(reference_pos.dtype),
        newly_active, previous_active, reference_pos, weighted_sum, weight_sum,
    )

    # If a newly activated group has no older anchor, retain its spectral/grid
    # initialization rather than forcing several new nodes onto one coordinate.
    anchored_nodes = np.flatnonzero(newly_active & (weight_sum > 0.0))
    if len(anchored_nodes) > 0:
        pos[anchored_nodes] = (
            weighted_sum[anchored_nodes]
            / weight_sum[anchored_nodes, None]
        ).astype(np.float32)
        draw = np.random.normal if rng is None else rng.normal
        pos[anchored_nodes] += draw(
            0.0, 0.05, (len(anchored_nodes), pos.shape[1])
        ).astype(np.float32)

    return active_mask


# A stage may use at most this fraction of the largest stable DT that its
# spring degree bound allows. The bound is never below the largest eigenvalue
# of the springs' Laplacian and, on SSN clusters, about twice it, so a lowered
# DT lands at 0.60 to 0.85 of the stage's true stability limit.
_DT_GUARD_FRACTION = 0.85


@_compile_serial_helper
def _spring_degree_bound(n_nodes, edges):
    """Largest d_u + d_v over the springs, d being a node's spring count.

    No eigenvalue of the springs' graph Laplacian exceeds it (Anderson and
    Morley). Self-loops are skipped because they exert no force.
    """
    degree = np.zeros(n_nodes, dtype=np.int64)
    for edge in range(edges.shape[0]):
        u = edges[edge, 0]
        v = edges[edge, 1]
        if u != v:
            degree[u] += 1
            degree[v] += 1
    bound = 0
    for edge in range(edges.shape[0]):
        u = edges[edge, 0]
        v = edges[edge, 1]
        if u != v and degree[u] + degree[v] > bound:
            bound = degree[u] + degree[v]
    return bound


def _euler_dt_limit(stiffness, damping):
    """Largest DT at which the damped semi-implicit Euler update keeps a
    spring mode of this stiffness from growing.

    The update is stable while stiffness * DT**2 < 4 - 2 * damping * DT.
    """
    return (math.sqrt(damping * damping + 4.0 * stiffness) - damping) / stiffness


# AUTO_DT runs each stage at this fraction of the limit for its springs plus
# one maximally stiff repulsive contact. Measured on the production kernel:
# paths, rings, trees, grids, stars, complete bipartite graphs, cliques of up
# to 60 nodes and pairs and triangles all settle at 0.5, while some jitter at
# 0.6 or above. Larger near-cliques may jitter slightly at any fraction above
# about 0.35.
_AUTO_DT_FRACTION = 0.5


def _steepest_repulsion(params):
    """Largest radial stiffness of one repulsive pair, reached where the force
    cap starts (or at the kernels' 0.5 distance floor without a cap)."""
    k_coul = float(params.get('COULOMB_K', 50.0))
    if k_coul <= 0.0:
        return 0.0
    max_f = float(params.get('MAX_FORCE_LIMIT', 20.0))
    steepest_distance = 0.5
    if max_f > 0.0:
        steepest_distance = max(steepest_distance, math.sqrt(k_coul / max_f))
    return 2.0 * k_coul / steepest_distance ** 3


def _three_figures(dt):
    """Round a chosen DT to three significant figures, so the log shows the
    exact DT that runs."""
    return float(f"{dt:.3g}")


def _stage_params(n_nodes, edges, params):
    """Return the parameters for one stage, choosing or capping its DT.

    With AUTO_DT, the stage runs at _AUTO_DT_FRACTION of the stability limit
    for its springs plus one maximally stiff repulsive contact, and the
    step-based stopping rules apply as configured.

    Otherwise the configured DT runs unless it would let the springs blow up:
    past their stability limit, zero-rest-length springs make every step
    overshoot, and nodes end up bouncing between the walls of the box. A
    stage that is safe at the configured DT gets `params` itself, so its
    layout stays bit-identical. Otherwise DT is lowered, and the step-based
    stopping rules are scaled so that the stage keeps its simulated time and
    the speed it treats as converged.
    """
    spring_stiffness = float(params.get('SPRING_K', 0.1)) * _spring_degree_bound(
        n_nodes, edges
    )
    damping = float(params.get('DAMPING', 0.5))
    if params.get('AUTO_DT', False):
        stiffness = spring_stiffness + 2.0 * _steepest_repulsion(params)
        if stiffness <= 0.0:
            return params
        stage = dict(params)
        stage['DT'] = _three_figures(
            _AUTO_DT_FRACTION * _euler_dt_limit(stiffness, damping)
        )
        print(f"  > Auto DT {stage['DT']:g} for this stage")
        return stage

    user_dt = float(params.get('DT', 0.1))
    if spring_stiffness <= 0.0:
        return params
    limit = _euler_dt_limit(spring_stiffness, damping)
    stage_dt = _three_figures(_DT_GUARD_FRACTION * limit)
    if stage_dt >= user_dt:
        return params

    scale = user_dt / stage_dt
    stage = dict(params)
    stage['DT'] = stage_dt
    stage['MAX_STEPS'] = int(math.ceil(params.get('MAX_STEPS', 2000) * scale))
    stage['RMSD_WINDOW'] = int(math.ceil(params.get('RMSD_WINDOW', 50) * scale))
    stage['RMSD_THRESHOLD'] = params.get('RMSD_THRESHOLD', 0.005) / scale
    print(
        f"  > DT {user_dt:g} -> {stage_dt:g} for this stage so its springs stay "
        f"stable (MAX_STEPS {stage['MAX_STEPS']}, RMSD_WINDOW "
        f"{stage['RMSD_WINDOW']} and RMSD_THRESHOLD "
        f"{stage['RMSD_THRESHOLD']:.3g} keep its simulated time and speed)"
    )
    return stage


def _report_boundary_contact(positions, box_limits, active_mask):
    """Warn when active nodes ended a stage on the wall of their box.

    A relaxed component sits well inside its box. Nodes end on a wall when
    the stage diverged, or when a long, sparse component needs more room.
    """
    positions = np.asarray(positions)
    limits = _normalize_box_limits(box_limits, len(positions))
    active = _normalize_active_mask(active_mask, len(positions))
    on_wall = active & (np.abs(positions) >= limits[:, None]).any(axis=1)
    count = int(np.count_nonzero(on_wall))
    if count:
        print(
            f"    - WARNING: {count} of {int(np.count_nonzero(active))} active "
            "nodes ended on the layout boundary. The stage diverged (lower DT) "
            "or the component needs more room (raise BOX_SCALE)."
        )


# A windowed RMSD that rises by more than this percentage is growing. Smaller
# rises are noise around a plateau: healthy stages have stopped on 0.1%.
_RISING_RMSD_PERCENT = 1.0


def _run_layout_stage(
    candidate,
    positions,
    edges,
    component_labels,
    box_limits,
    params,
    active_mask,
):
    """Run one serially dependent production stage on an explicit device."""
    if candidate.is_cpu:
        simulation = SSNSimulationCPU(
            positions.copy(), edges, component_labels, box_limits, params,
            active_mask=active_mask,
        )
    else:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch accelerator simulation is unavailable")
        simulation = SSNSimulationGPU(
            positions.copy(), edges, component_labels, box_limits, params,
            active_mask=active_mask, device=candidate.device,
        )

    try:
        rmsd_window = params.get('RMSD_WINDOW', 50)
        max_steps = params.get('MAX_STEPS', 2000)
        rmsd_buffer = deque(maxlen=rmsd_window)
        average_history = []
        average_rmsd = None

        for step in range(max_steps):
            rmsd = simulation.step(step)
            rmsd_buffer.append(rmsd)
            average_rmsd = np.mean(rmsd_buffer)

            if step > 0 and step % 500 == 0:
                print(
                    f"    - Step {step:04d}/{max_steps}: "
                    f"RMSD = {average_rmsd:.5f}"
                )

            if len(rmsd_buffer) == rmsd_window:
                average_history.append(average_rmsd)
                if average_rmsd < params.get('RMSD_THRESHOLD', 0.005):
                    print(
                        f"    - Converged at Step {step} "
                        f"(RMSD: {average_rmsd:.5f})"
                    )
                    break

                percentage_threshold = params.get(
                    'PERCENTAGE_DROP_THRESHOLD', 0.0
                )
                minimum_observation_steps = max_steps / 4.0
                trend_window = 10
                if (
                    percentage_threshold > 0.0
                    and len(average_history) >= (rmsd_window + trend_window)
                    and step > minimum_observation_steps
                ):
                    current_trend = np.mean(average_history[-trend_window:])
                    old_trend = np.mean(
                        average_history[
                            -(rmsd_window + trend_window):-rmsd_window
                        ]
                    )
                    if old_trend > 0:
                        percentage_drop = (
                            (old_trend - current_trend) / old_trend
                        ) * 100.0
                        if percentage_drop < -_RISING_RMSD_PERCENT:
                            print(
                                f"    - Not settling at Step {step}: RMSD rose "
                                f"{-percentage_drop:.3f}% over the last "
                                f"{rmsd_window} steps (RMSD: {average_rmsd:.5f})"
                            )
                            break
                        if percentage_drop < percentage_threshold:
                            print(
                                f"    - Plateau Reached at Step {step} "
                                f"(Drop: {percentage_drop:.3f}% < "
                                f"{percentage_threshold}%)"
                            )
                            break
        else:
            if average_rmsd is not None:
                print(
                    f"    - Step limit reached after {max_steps} steps "
                    f"(RMSD: {average_rmsd:.5f})"
                )

        final_positions = simulation.get_pos()
        _report_boundary_contact(final_positions, box_limits, active_mask)
        return final_positions
    finally:
        del simulation
        Hardware_Utils.release_device_cache(candidate)


def calculate_layout(connectivity, n_nodes, params):
    """
    Main layout generation pipeline.
    
    connectivity: N x 3 NumPy array representing [Source_Index, Target_Index, Score]
    n_nodes: Total number of nodes in the network
    params: Dictionary containing physics and execution parameters
    
    Returns:
        pos (np.ndarray): Final coordinates, (n_nodes, LAYOUT_DIMENSIONS)
        box_limit (float): Boundary box size
    """
    device_selection = params.get('LAYOUT_DEVICE_SELECTION', 'auto')
    dimensions = _resolve_dimensions(params)
    layout_seed = _resolve_seed(params)
    
    edges = connectivity[:, :2].astype(np.int32)
    edge_scores = connectivity[:, 2]
    
    # Initialize basic grid positioning to start
    base_box = np.sqrt(n_nodes) * 2.5 + 5.0
    initial_box_limit = base_box * params.get('BOX_SCALE', 1.0)
    initial_pos = _initial_lattice(n_nodes, initial_box_limit, dimensions)

    components = find_connected_components(n_nodes, edges)
    
    # 1. Sort components from largest to smallest
    components.sort(key=len, reverse=True)
    
    # 2. Skip single nodes completely
    active_comps = [c for c in components if len(c) > 1]
    singletons = len(components) - len(active_comps)
    
    large_comps = [c for c in active_comps if len(c) >= 500]
    small_comps = [c for c in active_comps if len(c) < 500]

    batches = []
    current_batch = []
    current_nodes = 0
    BATCH_LIMIT = 2000

    for comp in small_comps:
        if current_nodes + len(comp) > BATCH_LIMIT and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_nodes = 0
        current_batch.append(comp)
        current_nodes += len(comp)
    if current_batch:
        batches.append(current_batch)

    jobs = [[c] for c in large_comps] + batches
    
    print(f"Found {len(active_comps)} active components.")
    print(f"  > Simulating {len(large_comps)} massive components individually.")
    print(f"  > Grouped {len(small_comps)} small components into {len(batches)} parallel batches (Max {BATCH_LIMIT} nodes/batch).")
    print(f"  > Skipped {singletons} single nodes.")
    
    final_pos = np.copy(initial_pos)
    
    # Group edges and scores by component once, keeping edge-list order. A
    # node outside every active component (a singleton) maps to -1, so its
    # self-loops are dropped.
    node_to_comp_idx = np.full(n_nodes, -1, dtype=np.int64)
    for c_idx, comp in enumerate(active_comps):
        node_to_comp_idx[comp] = c_idx
    edge_order, edge_bounds = _stable_group_order(
        node_to_comp_idx[edges[:, 0]], len(active_comps)
    )
    comp_edges = {}
    comp_scores = {}
    for c_idx in range(len(active_comps)):
        members = edge_order[edge_bounds[c_idx]:edge_bounds[c_idx + 1]]
        comp_edges[c_idx] = edges[members]
        comp_scores[c_idx] = edge_scores[members]
    del edge_order

    device_rankings = Layout_Hardware.manual_layout_rankings(
        jobs, device_selection
    )
    if device_rankings is None:
        representative_indices = Layout_Hardware.representative_job_indices(
            jobs,
            node_to_comp_idx,
            comp_edges,
        )
        representative_batches = Layout_Hardware.prepare_representative_batches(
            jobs,
            representative_indices,
            node_to_comp_idx,
            comp_edges,
            comp_scores,
            params,
        )
        gpu_simulation_class = SSNSimulationGPU if HAS_TORCH else None
        device_rankings = {
            size_class: Layout_Hardware.benchmark_layout_devices(
                prepared,
                params,
                selection=device_selection,
                size_class=size_class,
                engine_label="SSN",
                cpu_simulation_class=SSNSimulationCPU,
                gpu_simulation_class=gpu_simulation_class,
            )
            for size_class, prepared in representative_batches.items()
        }

    if any(plans[0].candidate.is_cpu for plans in device_rankings.values()):
        print(
            f"CPU layout physics: {Numba_Threads.default_thread_count()} "
            f"threads on {Numba_Threads.usable_cpu_count()} logical CPUs."
        )

    # 3. Simulate jobs sequentially
    for job_idx, batch_comps in enumerate(jobs):
        job_rng = _job_generator(layout_seed, job_idx)
        prepared_batch = Layout_Hardware.prepare_layout_batch(
            batch_comps,
            node_to_comp_idx,
            comp_edges,
            comp_scores,
            params,
            rng=job_rng,
        )
        n_batch_nodes = prepared_batch.node_count
        is_large_job = prepared_batch.is_large_job
        batch_global_nodes = prepared_batch.global_nodes
        batch_edges = np.asarray(prepared_batch.edges).reshape(-1, 2)
        batch_scores = np.asarray(prepared_batch.scores, dtype=np.float64)
        batch_pos = prepared_batch.positions
        batch_comp_labels = prepared_batch.component_labels
        batch_box_limits = prepared_batch.box_limits
        size_class = Layout_Hardware.layout_size_class(n_batch_nodes)
        ranked_plans = device_rankings[size_class]

        if is_large_job:
             print(f"\nSimulating Large Component {job_idx+1}/{len(jobs)} ({n_batch_nodes} nodes)...")
        else:
             print(f"\nSimulating Batch {job_idx+1}/{len(jobs)} ({len(batch_comps)} components, {n_batch_nodes} nodes)...")

        cutoffs = [params.get('SIMILARITY_THRESHOLD', 0.0)]
        if is_large_job and params.get('ENABLE_PROGRESSIVE_SIMULATION', True) and n_batch_nodes > 2000 and len(batch_scores) > 10:
            sorted_local = np.sort(batch_scores)[::-1]
            n_edges = len(sorted_local)
            fractions = [0.2, 0.4, 0.6, 0.8]
            indices = [max(0, min(int(n_edges * f) - 1, n_edges - 1)) for f in fractions]
            raw_cutoffs = [sorted_local[i] for i in indices]
            
            cutoffs = []
            for c in raw_cutoffs:
                if not cutoffs or c < cutoffs[-1]:
                    cutoffs.append(c)
                    
            if not cutoffs or cutoffs[-1] > params.get('SIMILARITY_THRESHOLD', 0.0):
                cutoffs.append(params.get('SIMILARITY_THRESHOLD', 0.0))
            else:
                cutoffs[-1] = params.get('SIMILARITY_THRESHOLD', 0.0)
                
            print(f"  > Massive component detected. Using {len(cutoffs)}-stage progressive annealing (Edge-based).")
        
        previous_active = np.zeros(n_batch_nodes, dtype=np.bool_)

        for stage, cutoff in enumerate(cutoffs):
            in_stage = batch_scores >= cutoff
            if len(cutoffs) > 1:
                stage_edge_count = int(np.count_nonzero(in_stage))
                print(f"  > Stage {stage+1}/{len(cutoffs)}: Cutoff = {cutoff:.3f} | Active Edges: {stage_edge_count}")

            stage_edges = batch_edges[in_stage]
            stage_scores = batch_scores[in_stage]
            del in_stage
            stage_active_mask = _prepare_progressive_stage(
                batch_pos,
                stage_edges,
                stage_scores,
                previous_active,
                rng=job_rng,
            )
            previous_active = stage_active_mask.copy()

            local_edges = stage_edges.astype(np.int32)
            del stage_edges, stage_scores
            stage_params = _stage_params(n_batch_nodes, local_edges, params)

            stage_input = batch_pos.copy()
            failures = []
            for ranked_plan in ranked_plans:
                candidate = ranked_plan.candidate
                try:
                    batch_pos = _run_layout_stage(
                        candidate,
                        stage_input,
                        local_edges,
                        batch_comp_labels,
                        batch_box_limits,
                        stage_params,
                        stage_active_mask,
                    )
                    break
                except (RuntimeError, MemoryError, NotImplementedError) as error:
                    failures.append(f"{candidate.spec}: {error}")
                    if Hardware_Utils.normalize_device_selection(device_selection) != 'auto':
                        raise RuntimeError(
                            f"Layout failed on manually selected device "
                            f"'{candidate.spec}': {error}"
                        ) from error
                    print(
                        f"  > Layout stage failed on {candidate.display_name}: "
                        f"{error}. Retrying from saved stage input."
                    )
            else:
                raise RuntimeError(
                    "Layout stage failed on every ranked device: "
                    + "; ".join(failures)
                )
                
        # Update the final positions
        final_pos[batch_global_nodes] = batch_pos
            
    print("\nSimulation Complete.")
    
    # Pack independent components into a grid
    if dimensions == 3:
        final_pos, final_box_limit = pack_components_to_shells(
            final_pos, edges, n_nodes,
            params.get('PACKING_GRID_SIZE', 10.0),
            params.get('PACKING_PADDING', 50.0),
        )
        return final_pos, final_box_limit

    final_pos, final_box_limit = pack_components_to_grid(
        final_pos, edges, n_nodes, 
        params.get('PACKING_GRID_SIZE', 10.0), 
        params.get('PACKING_PADDING', 50.0),
        params.get('PACKING_GEOMETRY', 'Square')
    )
    
    return final_pos, final_box_limit
