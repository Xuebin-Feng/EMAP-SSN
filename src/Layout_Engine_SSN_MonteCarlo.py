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
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from utilities import Hardware_Utils, Layout_Hardware
import torch

HAS_TORCH = True

# --- SGLD / Monte Carlo Parameters (Defaults defined here as requested) ---
SGLD_NEGATIVE_SAMPLES = 20     # Number of random negative (repulsive) pairs sampled per node
SGLD_NOISE_SCALE = 1.0         # Global scalar for thermal noise perturbation
SGLD_START_TEMP = 1.5          # Starting SGLD temperature (kinetic heat) in Stage 1


# --- 1. Physics Kernels ---

def _get_physics_kernel_sgld():
    def _run_physics_kernel_sgld(pos, vel, springs, comp_labels, active_mask, active_nodes, active_comp_starts, active_comp_sizes, box_limits, dt, damping, k_spr, k_coul, max_f, max_total_repulsion, temperature, neg_samples_K, cutoff_dist):
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
            
            # Force gradient: -k_spr * distance * direction_vector
            acc[idx_a, 0] -= k_spr * dx
            acc[idx_a, 1] -= k_spr * dy
            acc[idx_b, 0] += k_spr * dx
            acc[idx_b, 1] += k_spr * dy
            
        # --- REPULSION (Negative Sampling) ---
        if n_balls > 1 and neg_samples_K > 0:
            cutoff_sq = cutoff_dist * cutoff_dist
            taper_start = cutoff_dist * 0.8
            taper_width = max(cutoff_dist * 0.2, 1e-9)
            for active_idx in range(active_nodes.shape[0]):
                i = active_nodes[active_idx]
                comp_idx = comp_labels[i]
                comp_start = active_comp_starts[comp_idx]
                comp_size = active_comp_sizes[comp_idx]
                if comp_size <= 1:
                    continue

                local_i = active_idx - comp_start
                scale_factor = (comp_size - 1.0) / neg_samples_K
                rep_x = 0.0
                rep_y = 0.0
                for k in range(neg_samples_K):
                    # Uniformly sample another node from this component. Mapping
                    # around local_i excludes self-samples without rejection.
                    local_j = np.random.randint(0, comp_size - 1)
                    if local_j >= local_i:
                        local_j += 1
                    j = active_nodes[comp_start + local_j]
                        
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist_sq = dx*dx + dy*dy
                    
                    if dist_sq > cutoff_sq:
                        continue
                        
                    dist = math.sqrt(dist_sq) + 1e-9
                    safe_dist = max(dist, 0.5)
                    f = k_coul / (safe_dist * safe_dist)
                    if max_f > 0.0 and f > max_f:
                        f = max_f
                    if dist > taper_start:
                        f *= max(0.0, (cutoff_dist - dist) / taper_width)
                        
                    rep_x += f * (dx / dist)
                    rep_y += f * (dy / dist)
                repulsion[i, 0] += rep_x * scale_factor
                repulsion[i, 1] += rep_y * scale_factor

        for active_idx in range(active_nodes.shape[0]):
            i = active_nodes[active_idx]
            rep_norm = math.sqrt(
                repulsion[i, 0] * repulsion[i, 0]
                + repulsion[i, 1] * repulsion[i, 1]
            )
            if max_total_repulsion > 0.0 and rep_norm > max_total_repulsion:
                rep_scale = max_total_repulsion / rep_norm
                repulsion[i, 0] *= rep_scale
                repulsion[i, 1] *= rep_scale
            acc[i, 0] += repulsion[i, 0]
            acc[i, 1] += repulsion[i, 1]
                    
        # --- INTEGRATION (Underdamped Langevin Dynamics) ---
        rmsd = 0.0
        # Thermal velocity noise scale: sqrt(2 * damping * temperature * dt)
        noise_scale = math.sqrt(2.0 * damping * temperature * dt) if temperature > 0.0 else 0.0
        
        n_active = active_nodes.shape[0]
        for i in range(n_balls):
            if not active_mask[i]:
                continue
            box_limit = box_limits[i]
            # Friction / Damping
            acc[i, 0] -= damping * vel[i, 0]
            acc[i, 1] -= damping * vel[i, 1]
            
            # Velocity update
            vel[i, 0] += acc[i, 0] * dt
            vel[i, 1] += acc[i, 1] * dt
            
            # Inject thermal kinetic noise (MCMC step)
            if noise_scale > 0.0:
                vel[i, 0] += np.random.normal(0, 1.0) * noise_scale
                vel[i, 1] += np.random.normal(0, 1.0) * noise_scale
                
            old_x, old_y = pos[i, 0], pos[i, 1]
            
            pos[i, 0] += vel[i, 0] * dt
            pos[i, 1] += vel[i, 1] * dt
            
            # Bouncing walls
            if pos[i, 0] > box_limit:
                pos[i, 0] = box_limit
                vel[i, 0] *= -0.5
            elif pos[i, 0] < -box_limit:
                pos[i, 0] = -box_limit
                vel[i, 0] *= -0.5
                
            if pos[i, 1] > box_limit:
                pos[i, 1] = box_limit
                vel[i, 1] *= -0.5
            elif pos[i, 1] < -box_limit:
                pos[i, 1] = -box_limit
                vel[i, 1] *= -0.5
                
            diff_x = pos[i, 0] - old_x
            diff_y = pos[i, 1] - old_y
            rmsd += diff_x*diff_x + diff_y*diff_y
            
        if n_active == 0:
            return 0.0
        return math.sqrt(rmsd / n_active)

    if NUMBA_AVAILABLE:
        return jit(nopython=True, fastmath=True)(_run_physics_kernel_sgld)
    return _run_physics_kernel_sgld

run_physics_kernel_sgld = _get_physics_kernel_sgld()


def _get_component_ranges(comp_labels):
    """Return contiguous component starts and sizes for a flattened batch."""
    labels = np.asarray(comp_labels, dtype=np.int32)
    if labels.size == 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty

    starts = np.flatnonzero(
        np.concatenate(([True], labels[1:] != labels[:-1]))
    ).astype(np.int32)
    sizes = np.diff(
        np.concatenate((starts, np.array([labels.size], dtype=np.int32)))
    ).astype(np.int32)

    run_labels = labels[starts]
    if starts.size != np.unique(labels).size:
        raise ValueError("Nodes belonging to a component must be contiguous in a simulation batch.")
    if not np.array_equal(run_labels, np.arange(starts.size, dtype=np.int32)):
        raise ValueError("Component labels must be dense integers starting at zero.")

    return starts, sizes


def _get_active_component_ranges(comp_labels, active_mask):
    """Group active node indices into contiguous per-component slices."""
    comp_starts, comp_sizes = _get_component_ranges(comp_labels)
    active_groups = []
    active_starts = np.zeros(len(comp_starts), dtype=np.int32)
    active_sizes = np.zeros(len(comp_starts), dtype=np.int32)
    cursor = 0

    for comp_idx, (start, size) in enumerate(zip(comp_starts, comp_sizes)):
        nodes = np.flatnonzero(active_mask[start:start + size]).astype(np.int32) + start
        active_starts[comp_idx] = cursor
        active_sizes[comp_idx] = len(nodes)
        cursor += len(nodes)
        if len(nodes) > 0:
            active_groups.append(nodes)

    if active_groups:
        active_nodes = np.concatenate(active_groups).astype(np.int32)
    else:
        active_nodes = np.zeros(0, dtype=np.int32)

    return active_nodes, active_starts, active_sizes


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
        (
            self.active_nodes,
            self.active_comp_starts,
            self.active_comp_sizes,
        ) = _get_active_component_ranges(self.comp_labels, self.active_mask)
        self.box_limits = _normalize_box_limits(box_limit, len(self.pos))
        self.params = params
        self.last_rmsd = 0.0
        
    def step(self, current_step):
        max_steps = self.params.get('MAX_STEPS', 2000)
        
        start_temp = self.params.get('SGLD_START_TEMP', SGLD_START_TEMP)
        noise_scale = self.params.get('SGLD_NOISE_SCALE', SGLD_NOISE_SCALE)
        
        # Thermal annealing is independent of the force model: the full
        # Coulomb cutoff and strength are active from the first step.
        progress = current_step / max(1.0, float(max_steps))
        if progress < 0.5:
            temperature = start_temp * (1.0 - progress / 0.5)
        else:
            temperature = 0.0
            
        temperature = max(0.0, temperature) * noise_scale
        
        # Calculate RMSD
        sgld_k = self.params.get('SGLD_K', SGLD_NEGATIVE_SAMPLES)
        cutoff_dist = self.params.get('COULOMB_CUTOFF', 30.0)
        self.last_rmsd = run_physics_kernel_sgld(
            self.pos, self.vel, self.springs,
            self.comp_labels, self.active_mask, self.active_nodes,
            self.active_comp_starts, self.active_comp_sizes, self.box_limits,
            self.params.get('DT', 0.1),
            self.params.get('DAMPING', 0.5),
            self.params.get('SPRING_K', 0.1),
            self.params.get('COULOMB_K', 50.0),
            self.params.get('MAX_FORCE_LIMIT', 20.0),
            self.params.get('MAX_TOTAL_REPULSION_FORCE', 0.0),
            temperature,
            sgld_k,
            cutoff_dist
        )
        return self.last_rmsd
        
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
            active_mask_np = _normalize_active_mask(active_mask, len(pos))
            active_nodes, active_starts, active_sizes = _get_active_component_ranges(
                comp_labels, active_mask_np
            )
            self.active_mask = torch.tensor(
                active_mask_np, dtype=torch.bool, device=self.device
            )
            self.active_nodes = torch.tensor(
                active_nodes, dtype=torch.long, device=self.device
            )
            self.n_active = len(active_nodes)
            self.active_comp_starts = torch.tensor(
                active_starts, dtype=torch.long, device=self.device
            )
            self.active_comp_sizes = torch.tensor(
                active_sizes, dtype=torch.long, device=self.device
            )
            self.box_limits = torch.tensor(
                _normalize_box_limits(box_limit, len(pos)),
                dtype=torch.float32,
                device=self.device
            )
            self.params = params
            self.last_rmsd = 0.0
        
        @torch.no_grad()
        def step(self, current_step):
            max_steps = self.params.get('MAX_STEPS', 2000)
            
            start_temp = self.params.get('SGLD_START_TEMP', SGLD_START_TEMP)
            noise_scale = self.params.get('SGLD_NOISE_SCALE', SGLD_NOISE_SCALE)
            
            # Thermal annealing does not delay the full repulsive force.
            progress = current_step / max(1.0, float(max_steps))
            if progress < 0.5:
                temperature = start_temp * (1.0 - progress / 0.5)
            else:
                temperature = 0.0
                
            temperature = max(0.0, temperature) * noise_scale
            
            acc = torch.zeros_like(self.pos)
            N = self.pos.size(0)
            
            # --- 2. ATTRACTION (Spring Forces) ---
            if len(self.springs) > 0:
                idx_a, idx_b = self.springs[:, 0], self.springs[:, 1]
                spring_active = self.active_mask[idx_a] & self.active_mask[idx_b]
                idx_a = idx_a[spring_active]
                idx_b = idx_b[spring_active]
                pa, pb = self.pos[idx_a], self.pos[idx_b]
                
                spring_k = self.params.get('SPRING_K', 0.1)
                fv = -spring_k * (pa - pb)
                
                acc.index_add_(0, idx_a, fv)
                acc.index_add_(0, idx_b, -fv)
                
                del pa, pb, fv
                
            # --- 3. REPULSION (Negative Sampling on GPU) ---
            k_coul = self.params.get('COULOMB_K', 50.0)
            max_f = self.params.get('MAX_FORCE_LIMIT', 20.0)
            cutoff_dist = self.params.get('COULOMB_CUTOFF', 30.0)
            
            sgld_k = self.params.get('SGLD_K', SGLD_NEGATIVE_SAMPLES)
            if len(self.active_nodes) > 1 and sgld_k > 0:
                source_nodes = self.active_nodes
                source_labels = self.comp_labels[source_nodes]
                node_starts = self.active_comp_starts[source_labels]
                node_sizes = self.active_comp_sizes[source_labels]
                eligible = node_sizes > 1

                # Draw from [0, component_size - 2], then map around the
                # current node's local index. This produces unbiased samples
                # from the other nodes in the same connected component.
                sample_span = (node_sizes - 1).clamp(min=1)
                neg_local = torch.floor(
                    torch.rand((len(source_nodes), sgld_k), device=self.device)
                    * sample_span.unsqueeze(1)
                ).long()
                local_i = (
                    torch.arange(len(source_nodes), device=self.device)
                    - node_starts
                )
                neg_local += (
                    (neg_local >= local_i.unsqueeze(1))
                    & eligible.unsqueeze(1)
                ).long()
                neg_nodes = self.active_nodes[
                    node_starts.unsqueeze(1) + neg_local
                ]
                
                # Reshape to compute pairwise distances
                pos_expanded = self.pos[source_nodes].unsqueeze(1)
                neg_pos = self.pos[neg_nodes]            # Shape: [N, sgld_k, 2]
                
                delta = pos_expanded - neg_pos           # Shape: [N, sgld_k, 2]
                dist = delta.norm(dim=2) + 1e-9          # Shape: [N, sgld_k]
                
                # Repulsion magnitude: f = k_coul / max(dist, 0.5)^2
                safe_dist = torch.clamp(dist, min=0.5)
                f_mag = k_coul / (safe_dist ** 2)
                if max_f > 0.0:
                    f_mag.clamp_(max=max_f)
                taper_start = cutoff_dist * 0.8
                taper_width = max(cutoff_dist * 0.2, 1e-9)
                taper = ((cutoff_dist - dist) / taper_width).clamp(
                    min=0.0, max=1.0
                )
                taper = torch.where(dist > taper_start, taper, 1.0)
                f_mag *= taper
                
                # Mask singleton components and nodes beyond COULOMB_CUTOFF.
                is_far = dist > cutoff_dist
                cond = eligible.unsqueeze(1) & (~is_far)
                f_mag = torch.where(cond, f_mag, 0.0)
                
                # Force vector
                f_vec = (f_mag / dist).unsqueeze(2) * delta  # Shape: [N, sgld_k, 2]
                
                # Scale each component's estimator by its own population.
                scale_factor = (node_sizes - 1).to(self.pos.dtype) / float(sgld_k)
                sampled_repulsion = f_vec.sum(dim=1) * scale_factor.unsqueeze(1)
                max_total_repulsion = self.params.get(
                    'MAX_TOTAL_REPULSION_FORCE', 0.0
                )
                if max_total_repulsion > 0.0:
                    repulsion_norm = sampled_repulsion.norm(dim=1, keepdim=True)
                    repulsion_scale = (
                        max_total_repulsion
                        / repulsion_norm.clamp(min=1e-12)
                    ).clamp(max=1.0)
                    sampled_repulsion *= repulsion_scale
                acc.index_add_(0, source_nodes, sampled_repulsion)
                
                del source_nodes, source_labels, node_starts, node_sizes
                del eligible, sample_span, neg_local
                del local_i, neg_nodes, pos_expanded, neg_pos, delta, dist
                del safe_dist, f_mag, f_vec, is_far, cond, scale_factor, taper
                del sampled_repulsion
                if max_total_repulsion > 0.0:
                    del repulsion_norm, repulsion_scale
                
            # --- 4. INTEGRATION (Underdamped Langevin Dynamics) ---
            damping = self.params.get('DAMPING', 0.5)
            dt = self.params.get('DT', 0.1)
            
            # Apply friction
            acc -= damping * self.vel
            acc[~self.active_mask] = 0.0
            self.vel[~self.active_mask] = 0.0
            self.vel[self.active_mask] += acc[self.active_mask] * dt
            
            # Add thermal Langevin noise to velocity
            if temperature > 0.0:
                noise_scale = math.sqrt(2.0 * damping * temperature * dt)
                noise = (
                    torch.randn(
                        (self.n_active, 2),
                        device=self.device,
                        dtype=self.vel.dtype,
                    )
                    * noise_scale
                )
                self.vel[self.active_mask] += noise
                
            old = self.pos.clone()
            self.pos[self.active_mask] += self.vel[self.active_mask] * dt
            
            # Boundary collisions (Bouncing)
            out_of_bounds_x = (self.pos[:, 0].abs() > self.box_limits) & self.active_mask
            out_of_bounds_y = (self.pos[:, 1].abs() > self.box_limits) & self.active_mask
            self.vel[out_of_bounds_x, 0] *= -0.5
            self.vel[out_of_bounds_y, 1] *= -0.5
            
            limits = self.box_limits[self.active_mask].unsqueeze(1)
            active_pos = self.pos[self.active_mask]
            self.pos[self.active_mask] = torch.maximum(
                torch.minimum(active_pos, limits),
                -limits
            )

            # Transfer RMSD to host once every 10 steps to prevent PCIe stalls
            if (current_step % 10 == 0) or (current_step == max_steps - 1):
                if self.active_mask.any():
                    self.last_rmsd = (
                        (self.pos[self.active_mask] - old[self.active_mask])
                        .norm(dim=1).pow(2).mean().sqrt().item()
                    )
                else:
                    self.last_rmsd = 0.0
                
            del acc, old, limits, active_pos
            if temperature > 0.0:
                del noise
            return self.last_rmsd

        def get_pos(self): return self.pos.cpu().numpy()


# --- Component-energy Monte Carlo optimizer ---

MC_SWEEPS = 250
MC_QUENCH_SWEEPS = 25
MC_TELEPORT_PROBABILITY = 0.10
MC_RANDOM_SEED = 42
_MIN_REPULSION_DISTANCE = 0.5
_TEMPERATURE_END_RATIO = 1.0e-3


def _repulsive_potential_scalar(distance, k_coul, max_force, cutoff):
    """Return the pair potential whose negative derivative is the MC force.

    The potential is anchored to zero at ``cutoff`` and exactly integrates the
    force law used by the physics engines: a finite-distance core, optional
    per-pair force cap, and the linear taper over the outer 20% of the cutoff.
    """
    if cutoff <= 0.0 or k_coul <= 0.0 or distance >= cutoff:
        return 0.0

    lower = max(float(distance), 0.0)
    taper_start = cutoff * 0.8
    taper_width = max(cutoff * 0.2, 1.0e-12)

    flat_radius = _MIN_REPULSION_DISTANCE
    if max_force > 0.0:
        cap_radius = math.sqrt(k_coul / max_force)
        if cap_radius > flat_radius:
            flat_radius = cap_radius
    flat_force = k_coul / (flat_radius * flat_radius)

    def integrate_flat(lo, hi):
        if hi <= lo:
            return 0.0
        value = 0.0
        inner_hi = min(hi, taper_start)
        if inner_hi > lo:
            value += flat_force * (inner_hi - lo)
        taper_lo = max(lo, taper_start)
        if hi > taper_lo:
            value += (flat_force / taper_width) * (
                cutoff * (hi - taper_lo)
                - 0.5 * (hi * hi - taper_lo * taper_lo)
            )
        return value

    def integrate_inverse_square(lo, hi):
        if hi <= lo:
            return 0.0
        value = 0.0
        inner_hi = min(hi, taper_start)
        if inner_hi > lo:
            value += k_coul * (1.0 / lo - 1.0 / inner_hi)
        taper_lo = max(lo, taper_start)
        if hi > taper_lo:
            value += (k_coul / taper_width) * (
                cutoff / taper_lo
                - cutoff / hi
                + math.log(taper_lo / hi)
            )
        return value

    energy = 0.0
    flat_hi = min(flat_radius, cutoff)
    if lower < flat_hi:
        energy += integrate_flat(lower, flat_hi)
    inverse_lo = max(lower, flat_radius)
    if inverse_lo < cutoff:
        energy += integrate_inverse_square(inverse_lo, cutoff)
    return energy


if NUMBA_AVAILABLE:
    repulsive_pair_energy = jit(
        nopython=True, fastmath=True
    )(_repulsive_potential_scalar)
else:
    repulsive_pair_energy = _repulsive_potential_scalar


def _repulsive_energy_sum_scalar(
    position,
    positions,
    candidate_indices,
    k_coul,
    max_force,
    cutoff,
):
    total = 0.0
    for offset in range(candidate_indices.shape[0]):
        other = candidate_indices[offset]
        dx = position[0] - positions[other, 0]
        dy = position[1] - positions[other, 1]
        distance = math.sqrt(dx * dx + dy * dy)
        total += repulsive_pair_energy(
            distance, k_coul, max_force, cutoff
        )
    return total


if NUMBA_AVAILABLE:
    repulsive_energy_sum = jit(
        nopython=True, fastmath=True
    )(_repulsive_energy_sum_scalar)
else:
    repulsive_energy_sum = _repulsive_energy_sum_scalar


class ComponentSpatialHash:
    """Exact cutoff-neighbor index for one independently optimized component."""

    def __init__(self, positions, cutoff):
        self.cutoff = float(cutoff)
        self.cells = {}
        self.node_cells = [None] * len(positions)
        if self.cutoff > 0.0:
            for node, position in enumerate(positions):
                self._insert(node, position)

    def _key(self, position):
        return (
            int(math.floor(float(position[0]) / self.cutoff)),
            int(math.floor(float(position[1]) / self.cutoff)),
        )

    def _insert(self, node, position):
        key = self._key(position)
        self.cells.setdefault(key, set()).add(int(node))
        self.node_cells[node] = key

    def candidates(self, position):
        if self.cutoff <= 0.0:
            return set()
        cell_x, cell_y = self._key(position)
        result = set()
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                result.update(
                    self.cells.get((cell_x + offset_x, cell_y + offset_y), ())
                )
        return result

    def update(self, node, position):
        if self.cutoff <= 0.0:
            return
        old_key = self.node_cells[node]
        new_key = self._key(position)
        if old_key == new_key:
            return
        old_nodes = self.cells[old_key]
        old_nodes.remove(node)
        if not old_nodes:
            del self.cells[old_key]
        self.cells.setdefault(new_key, set()).add(node)
        self.node_cells[node] = new_key


class ComponentEnergyMonteCarlo:
    """Metropolis annealing over the exact total energy of one component."""

    def __init__(self, positions, edges, box_limit, params, rng):
        _validate_energy_monte_carlo_params(params)
        try:
            position_array = np.asarray(positions)
        except (TypeError, ValueError) as error:
            raise ValueError("positions must be a numeric N x 2 array.") from error
        if position_array.ndim != 2 or position_array.shape[1] != 2:
            raise ValueError("positions must be a two-dimensional N x 2 array.")
        if position_array.shape[0] < 1:
            raise ValueError("A component must contain at least one position.")
        if not np.issubdtype(position_array.dtype, np.number) or np.issubdtype(
            position_array.dtype, np.complexfloating
        ):
            raise ValueError("positions must contain finite real numeric values.")
        try:
            finite_positions = np.isfinite(position_array).all()
        except TypeError as error:
            raise ValueError("positions must contain finite numeric values.") from error
        if not finite_positions:
            raise ValueError("positions must contain finite numeric values.")

        try:
            edge_array = np.asarray(edges)
        except (TypeError, ValueError) as error:
            raise ValueError("edges must be a numeric M x 2 array.") from error
        if edge_array.size == 0:
            edge_array = np.zeros((0, 2), dtype=np.int32)
        elif edge_array.ndim != 2 or edge_array.shape[1] != 2:
            raise ValueError("edges must be a two-dimensional M x 2 array.")
        if not np.issubdtype(edge_array.dtype, np.number) or np.issubdtype(
            edge_array.dtype, np.complexfloating
        ):
            raise ValueError("edges must contain finite integer node indices.")
        try:
            finite_edges = np.isfinite(edge_array).all()
        except TypeError as error:
            raise ValueError("edges must contain finite integer node indices.") from error
        if not finite_edges or not np.equal(edge_array, np.floor(edge_array)).all():
            raise ValueError("edges must contain finite integer node indices.")
        if edge_array.size and (
            np.min(edge_array) < 0 or np.max(edge_array) >= len(position_array)
        ):
            raise ValueError("edges contain a node index outside positions.")

        try:
            limit_array = np.asarray(box_limit, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "box_limit must contain finite positive values."
            ) from error
        if (
            limit_array.size == 0
            or not np.isfinite(limit_array).all()
            or np.any(limit_array <= 0.0)
        ):
            raise ValueError("box_limit must contain finite positive values.")
        if not np.all(limit_array == limit_array[0]):
            raise ValueError("One component must use one uniform box_limit.")

        self.pos = position_array.astype(np.float64, copy=True)
        self.edges = edge_array.astype(np.int32, copy=False)
        self.box_limit = float(limit_array[0])
        if np.any(np.abs(self.pos) > self.box_limit):
            raise ValueError("Initial component positions must lie within box_limit.")
        self.params = params
        self.rng = rng
        self.spring_k = float(params.get('SPRING_K', 0.1))
        self.k_coul = float(params.get('COULOMB_K', 50.0))
        self.cutoff = float(params.get('COULOMB_CUTOFF', 30.0))
        self.max_force = float(params.get('MAX_FORCE_LIMIT', 20.0))
        self.teleport_probability = float(
            params.get('MC_TELEPORT_PROBABILITY', MC_TELEPORT_PROBABILITY)
        )
        self.proposal_scale = (
            self.cutoff if self.cutoff > 0.0 else max(self.box_limit, 1.0)
        )
        self.local_sigma = 0.1 * self.proposal_scale
        self.teleport_sigma = 0.25 * self.proposal_scale
        self.incident_nodes = [[] for _ in range(len(self.pos))]
        graph_neighbor_sets = [set() for _ in range(len(self.pos))]
        for node_a, node_b in self.edges:
            if node_a == node_b:
                continue
            self.incident_nodes[int(node_a)].append(int(node_b))
            self.incident_nodes[int(node_b)].append(int(node_a))
            graph_neighbor_sets[int(node_a)].add(int(node_b))
            graph_neighbor_sets[int(node_b)].add(int(node_a))
        self.graph_neighbors = [
            np.asarray(sorted(neighbors), dtype=np.int32)
            for neighbors in graph_neighbor_sets
        ]
        self.spatial = ComponentSpatialHash(self.pos, self.cutoff)
        self.current_energy = self.total_energy()
        self.best_energy = self.current_energy
        self.best_pos = self.pos.copy()
        self.energy_history = [self.current_energy]
        self.max_energy_drift = 0.0
        self.quench_energy_history = []
        self.quench_sweeps_completed = 0

    def _pair_energy(self, position_a, position_b):
        distance = float(np.linalg.norm(position_a - position_b))
        return float(repulsive_pair_energy(
            distance, self.k_coul, self.max_force, self.cutoff
        ))

    def total_energy(self):
        if len(self.edges):
            spring_deltas = (
                self.pos[self.edges[:, 0]] - self.pos[self.edges[:, 1]]
            )
            spring_energy = 0.5 * self.spring_k * float(
                np.einsum('ij,ij->', spring_deltas, spring_deltas)
            )
        else:
            spring_energy = 0.0

        repulsive_energy = 0.0
        if self.cutoff > 0.0 and self.k_coul > 0.0:
            for node_a, position in enumerate(self.pos):
                candidate_indices = np.fromiter(
                    sorted(
                        node_b
                        for node_b in self.spatial.candidates(position)
                        if node_b > node_a
                    ),
                    dtype=np.int32,
                )
                repulsive_energy += repulsive_energy_sum(
                    position,
                    self.pos,
                    candidate_indices,
                    self.k_coul,
                    self.max_force,
                    self.cutoff,
                )
        return float(spring_energy + repulsive_energy)

    def energy_delta(self, node, proposed_position):
        old_position = self.pos[node]
        delta_energy = 0.0
        for other in self.incident_nodes[node]:
            old_delta = old_position - self.pos[other]
            new_delta = proposed_position - self.pos[other]
            delta_energy += 0.5 * self.spring_k * (
                float(np.dot(new_delta, new_delta))
                - float(np.dot(old_delta, old_delta))
            )

        candidates = self.spatial.candidates(old_position)
        candidates.update(self.spatial.candidates(proposed_position))
        candidate_indices = np.fromiter(
            sorted(other for other in candidates if other != node),
            dtype=np.int32,
        )
        delta_energy += repulsive_energy_sum(
            proposed_position,
            self.pos,
            candidate_indices,
            self.k_coul,
            self.max_force,
            self.cutoff,
        ) - repulsive_energy_sum(
            old_position,
            self.pos,
            candidate_indices,
            self.k_coul,
            self.max_force,
            self.cutoff,
        )
        return float(delta_energy)

    def _proposal(self, node, *, local_only=False):
        old_position = self.pos[node]
        if local_only or self.rng.random() >= self.teleport_probability:
            return (
                old_position + self.rng.normal(0.0, self.local_sigma, 2),
                0.0,
                'local',
            )

        if self.rng.random() < 0.8 and self.graph_neighbors[node].size:
            centroid = self.pos[self.graph_neighbors[node]].mean(axis=0)
            proposed = centroid + self.rng.normal(
                0.0, self.teleport_sigma, 2
            )
            inverse_variance = 1.0 / (self.teleport_sigma * self.teleport_sigma)
            old_sq = float(np.dot(old_position - centroid, old_position - centroid))
            new_sq = float(np.dot(proposed - centroid, proposed - centroid))
            log_hastings = -0.5 * (old_sq - new_sq) * inverse_variance
            return proposed, log_hastings, 'neighbor'

        proposed = self.rng.uniform(-self.box_limit, self.box_limit, 2)
        return proposed, 0.0, 'uniform'

    def _attempt(self, node, temperature, *, local_only=False):
        proposed, log_hastings, proposal_kind = self._proposal(
            node, local_only=local_only
        )
        if not np.isfinite(proposed).all() or np.any(
            np.abs(proposed) > self.box_limit
        ):
            return False, proposal_kind

        delta_energy = self.energy_delta(node, proposed)
        if not math.isfinite(delta_energy):
            return False, proposal_kind

        if temperature <= 0.0:
            accepted = delta_energy < 0.0
        else:
            log_acceptance = -delta_energy / temperature + log_hastings
            accepted = log_acceptance >= 0.0 or math.log(
                max(float(self.rng.random()), 1.0e-300)
            ) < log_acceptance

        if not accepted:
            return False, proposal_kind

        self.pos[node] = proposed
        self.spatial.update(node, proposed)
        self.current_energy += delta_energy
        if self.current_energy < self.best_energy:
            exact_energy = self.total_energy()
            drift = abs(exact_energy - self.current_energy)
            self.max_energy_drift = max(self.max_energy_drift, drift)
            self.current_energy = exact_energy
            if exact_energy < self.best_energy:
                self.best_energy = exact_energy
                self.best_pos = self.pos.copy()
        return True, proposal_kind

    def _calibrate_temperature(self):
        proposal_count = min(256, max(32, len(self.pos)))
        positive_deltas = []
        for _ in range(proposal_count):
            node = int(self.rng.integers(0, len(self.pos)))
            proposed, _, _ = self._proposal(node)
            if np.isfinite(proposed).all() and not np.any(
                np.abs(proposed) > self.box_limit
            ):
                delta_energy = self.energy_delta(node, proposed)
                if math.isfinite(delta_energy) and delta_energy > 0.0:
                    positive_deltas.append(delta_energy)
        if not positive_deltas:
            return 0.0
        return float(np.median(positive_deltas) / math.log(2.0))

    def _reconcile_energy(self):
        exact_energy = self.total_energy()
        drift = abs(exact_energy - self.current_energy)
        self.max_energy_drift = max(self.max_energy_drift, drift)
        self.current_energy = exact_energy
        self.best_energy = self.total_energy_for(self.best_pos)
        if exact_energy < self.best_energy:
            self.best_energy = exact_energy
            self.best_pos = self.pos.copy()
        self.energy_history.append(exact_energy)

    def optimize(self):
        sweeps = int(self.params.get('MC_SWEEPS', MC_SWEEPS))
        quench_sweeps = int(
            self.params.get('MC_QUENCH_SWEEPS', MC_QUENCH_SWEEPS)
        )
        start_temperature = self._calibrate_temperature()
        adaptation_sweeps = max(1, int(math.ceil(0.2 * sweeps)))
        local_attempts = 0
        local_accepts = 0

        for sweep in range(sweeps):
            if start_temperature > 0.0 and sweeps > 1:
                progress = sweep / float(sweeps - 1)
                temperature = start_temperature * (
                    _TEMPERATURE_END_RATIO ** progress
                )
            else:
                temperature = 0.0

            for node in self.rng.permutation(len(self.pos)):
                accepted, proposal_kind = self._attempt(
                    int(node), temperature
                )
                if proposal_kind == 'local':
                    local_attempts += 1
                    local_accepts += int(accepted)

            if (
                sweep < adaptation_sweeps
                and (sweep + 1) % 10 == 0
                and local_attempts > 0
            ):
                acceptance_rate = local_accepts / float(local_attempts)
                if acceptance_rate < 0.25:
                    self.local_sigma *= 0.8
                elif acceptance_rate > 0.45:
                    self.local_sigma *= 1.2
                self.local_sigma = min(
                    max(self.local_sigma, 0.01 * self.proposal_scale),
                    self.proposal_scale,
                )
                local_attempts = 0
                local_accepts = 0

            if (sweep + 1) % 10 == 0:
                self._reconcile_energy()

        quench_start_sigma = self.local_sigma
        inactive_sweeps = 0
        for sweep in range(quench_sweeps):
            if quench_sweeps > 1:
                progress = sweep / float(quench_sweeps - 1)
                self.local_sigma = quench_start_sigma * (0.1 ** progress)
            accepted_count = 0
            for node in self.rng.permutation(len(self.pos)):
                accepted, _ = self._attempt(
                    int(node), 0.0, local_only=True
                )
                accepted_count += int(accepted)
            self._reconcile_energy()
            self.quench_sweeps_completed = sweep + 1
            self.quench_energy_history.append(self.current_energy)
            if accepted_count == 0:
                inactive_sweeps += 1
                if inactive_sweeps >= 5:
                    break
            else:
                inactive_sweeps = 0

        self._reconcile_energy()
        returned_positions = self.best_pos.astype(np.float32)
        best_energy = self.total_energy_for(returned_positions)
        self.best_pos = returned_positions.astype(np.float64)
        self.best_energy = best_energy
        return returned_positions, float(best_energy)

    def total_energy_for(self, positions):
        current_pos = self.pos
        current_spatial = self.spatial
        try:
            self.pos = np.asarray(positions, dtype=np.float64)
            self.spatial = ComponentSpatialHash(self.pos, self.cutoff)
            return self.total_energy()
        finally:
            self.pos = current_pos
            self.spatial = current_spatial


# --- 2. Components & Packing Logic (Identical API for seamless integration) ---

def find_connected_components(n_nodes, edges):
    """Finds all independent subgraphs using Breadth-First Search."""
    adj = {i: [] for i in range(n_nodes)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    visited = np.zeros(n_nodes, dtype=bool)
    components = []
    
    for i in range(n_nodes):
        if not visited[i]:
            comp = []
            q = [i]
            visited[i] = True
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        q.append(neighbor)
            components.append(comp)
    return components


def get_component_labels(n_nodes, edges):
    """Maps each node to its connected component ID."""
    components = find_connected_components(n_nodes, edges)
    labels = np.zeros(n_nodes, dtype=np.int32)
    for c_id, comp in enumerate(components):
        for node in comp:
            labels[node] = c_id
    return labels


def pack_components_to_grid(pos, edges, n_nodes, grid_size, padding, packing_geometry="Square"):
    """Packs independent components into a strict grid layout."""
    components = find_connected_components(n_nodes, edges)
    if not components:
        return pos, 100.0

    node_to_comp = {}
    for c_id, comp in enumerate(components):
        for node in comp:
            node_to_comp[node] = c_id
            
    comp_edges = {c_id: [] for c_id in range(len(components))}
    for u, v in edges:
        c_id = node_to_comp.get(u)
        if c_id is not None:
            comp_edges[c_id].append((u, v))

    comp_info = []
    for c_id, comp in enumerate(components):
        idx = np.array(comp)
        comp_pos = pos[idx]
        
        min_x = np.min(comp_pos[:, 0])
        max_y = np.max(comp_pos[:, 1]) 
        
        shifted_pos = comp_pos - [min_x, max_y]
        global_to_local = {g: l for l, g in enumerate(comp)}
        points = list(shifted_pos)
        
        for u, v in comp_edges[c_id]:
            p1 = shifted_pos[global_to_local[u]]
            p2 = shifted_pos[global_to_local[v]]
            dist = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            steps = int(dist / (grid_size / 4)) + 1
            for i in range(1, steps):
                t = i / steps
                px = p1[0] + t * (p2[0] - p1[0])
                py = p1[1] + t * (p2[1] - p1[1])
                points.append([px, py])
                
        max_c, max_r = 0, 0
        pad = padding / 2.0
        
        for px, py in points:
            pos_y = -py
            c_max = int((px + pad) / grid_size)
            r_max = int((pos_y + pad) / grid_size)
            max_c = max(max_c, c_max)
            max_r = max(max_r, r_max)
            
        cols = max_c + 1
        rows = max_r + 1
        mask = np.zeros((rows, cols), dtype=bool)
        
        for px, py in points:
            pos_y = -py
            c_min = int(max(0, px - pad) / grid_size)
            c_max = int((px + pad) / grid_size)
            r_min = int(max(0, pos_y - pad) / grid_size)
            r_max = int((pos_y + pad) / grid_size)
            mask[r_min:r_max+1, c_min:c_max+1] = True
            
        comp_info.append({
            'indices': idx,
            'shifted_pos': shifted_pos,
            'mask': mask,
            'cols': cols,
            'rows': rows,
            'area': np.sum(mask),
            'num_nodes': len(idx)
        })
        
    comp_info.sort(key=lambda x: (x['area'], x['num_nodes']), reverse=True)
    
    # --- 3. Prepare the Master Global Grid and Run Spiral Placement ---
    total_area = sum(c['area'] for c in comp_info)
    max_cols = max(c['cols'] for c in comp_info)
    max_rows = max(c['rows'] for c in comp_info)
    
    is_circle = (packing_geometry.lower() == "circle")
    multiplier = 2.0 if is_circle else 1.5
    
    # Start with a grid size S estimated from total area, scaled to prevent border clipping
    S = max(int(math.ceil(math.sqrt(total_area) * multiplier)), max_cols, max_rows)
    
    new_pos = np.zeros((n_nodes, 2), dtype=np.float32)
    # Fill unconnected nodes first
    new_pos[:] = pos[:]
    
    # Center nodes aesthetically within their grid squares
    center_x_offset = grid_size / 2.0  
    center_y_offset = -grid_size / 2.0
    
    while True:
        grid_map = np.zeros((S, S), dtype=bool)
        center_r = S // 2
        center_c = S // 2
        
        # Calculate physical center coordinate
        if is_circle:
            center_x_phys = center_c + 0.5 * (center_r % 2)
            center_y_phys = -center_r * (math.sqrt(3.0) / 2.0)
        else:
            center_x_phys = center_c
            center_y_phys = -center_r
            
        # Generate all coordinates in the grid map
        coords = []
        for r in range(S):
            for c in range(S):
                if is_circle:
                    x_phys = c + 0.5 * (r % 2)
                    y_phys = -r * (math.sqrt(3.0) / 2.0)
                    dist = (x_phys - center_x_phys)**2 + (y_phys - center_y_phys)**2
                else:
                    dist_l_inf = max(abs(r - center_r), abs(c - center_c))
                    dist_l_2 = (r - center_r)**2 + (c - center_c)**2
                    dist = (dist_l_inf, dist_l_2)
                coords.append((r, c, dist))
                
        # Sort coords by distance from center (ascending)
        coords.sort(key=lambda x: x[2])
        
        success = True
        placed_offsets = []
        
        for comp in comp_info:
            mask = comp['mask']
            h, w = comp['rows'], comp['cols']
            placed = False
            
            for r_center, c_center, _ in coords:
                # Target top-left row/col so that component center aligns close to r_center, c_center
                r = r_center - h // 2
                c = c_center - w // 2
                
                if r >= 0 and r + h <= S and c >= 0 and c + w <= S:
                    if not np.any(grid_map[r:r+h, c:c+w] & mask):
                        grid_map[r:r+h, c:c+w] |= mask
                        placed_offsets.append((r, c))
                        placed = True
                        break
                        
            if not placed:
                success = False
                break
                
        if success:
            # Apply offsets
            for comp, (r, c) in zip(comp_info, placed_offsets):
                if is_circle:
                    # Hexagonal physical coordinates
                    x_offset = (c + 0.5 * (r % 2)) * grid_size
                    y_offset = -r * grid_size * (math.sqrt(3.0) / 2.0)
                else:
                    # Square grid physical coordinates
                    x_offset = c * grid_size
                    y_offset = -r * grid_size
                    
                new_pos[comp['indices'], 0] = comp['shifted_pos'][:, 0] + x_offset + center_x_offset
                new_pos[comp['indices'], 1] = comp['shifted_pos'][:, 1] + y_offset + center_y_offset
            break
        else:
            # Increase grid size and retry
            S = int(S * 1.1) + 2
                
    global_min = np.min(new_pos, axis=0)
    global_max = np.max(new_pos, axis=0)
    center = (global_max + global_min) / 2.0
    new_pos -= center
    
    new_box_limit = max(global_max[0] - global_min[0], global_max[1] - global_min[1]) / 2.0 * 1.1
    return new_pos, new_box_limit


# --- 3. Main Layout Entrypoint ---

def _prepare_progressive_stage(pos, stage_edges, stage_scores, previous_active):
    """Activate stage nodes and place newly introduced nodes near active neighbors."""
    active_mask = np.zeros(len(pos), dtype=np.bool_)
    for u, v in stage_edges:
        active_mask[u] = True
        active_mask[v] = True

    newly_active = active_mask & (~previous_active)
    if not np.any(newly_active) or not np.any(previous_active):
        return active_mask

    reference_pos = pos.copy()
    weighted_sum = np.zeros_like(pos, dtype=np.float64)
    weight_sum = np.zeros(len(pos), dtype=np.float64)

    for (u, v), score in zip(stage_edges, stage_scores):
        weight = max(float(score), 1e-9)
        if newly_active[u] and previous_active[v]:
            weighted_sum[u] += reference_pos[v] * weight
            weight_sum[u] += weight
        if newly_active[v] and previous_active[u]:
            weighted_sum[v] += reference_pos[u] * weight
            weight_sum[v] += weight

    anchored_nodes = np.flatnonzero(newly_active & (weight_sum > 0.0))
    if len(anchored_nodes) > 0:
        pos[anchored_nodes] = (
            weighted_sum[anchored_nodes]
            / weight_sum[anchored_nodes, None]
        ).astype(np.float32)
        pos[anchored_nodes] += np.random.normal(
            0.0, 0.05, (len(anchored_nodes), 2)
        ).astype(np.float32)

    return active_mask


def _run_layout_stage(
    candidate,
    positions,
    edges,
    component_labels,
    box_limits,
    params,
    active_mask,
):
    """Run one serially dependent SGLD production stage on a device."""
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

        for step in range(max_steps):
            rmsd = simulation.step(step)
            rmsd_buffer.append(rmsd)
            average_rmsd = np.mean(rmsd_buffer)

            if step > 0 and step % 500 == 0:
                print(
                    f"    - Step {step:04d}/{max_steps} | "
                    f"RMSD: {average_rmsd:.5f}"
                )

            if len(rmsd_buffer) == rmsd_window:
                average_history.append(average_rmsd)
                progress = step / float(max_steps)
                if (
                    average_rmsd < params.get('RMSD_THRESHOLD', 0.005)
                    and progress > 0.8
                ):
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
                        if (
                            percentage_drop < percentage_threshold
                            and progress > 0.8
                        ):
                            print(
                                f"    - Plateau Reached at Step {step} "
                                f"(Drop: {percentage_drop:.3f}% < "
                                f"{percentage_threshold}%)"
                            )
                            break

        return simulation.get_pos()
    finally:
        del simulation
        Hardware_Utils.release_device_cache(candidate)


def _calculate_layout_sgld_legacy(connectivity, n_nodes, params):
    """
    Main layout generation pipeline using Monte Carlo SGLD.
    """
    device_selection = params.get('LAYOUT_DEVICE_SELECTION', 'auto')
    
    edges = connectivity[:, :2].astype(np.int32)
    edge_scores = connectivity[:, 2]
    
    # Initialize basic grid positioning
    print("Computing initial node positions using Laplacian Spectral / Grid layouts...")
    side = int(np.ceil(np.sqrt(n_nodes)))
    base_box = np.sqrt(n_nodes) * 2.5 + 5.0
    initial_box_limit = base_box * params.get('BOX_SCALE', 1.0)
    x = np.linspace(-initial_box_limit*0.5, initial_box_limit*0.5, side)
    y = np.linspace(-initial_box_limit*0.5, initial_box_limit*0.5, side)
    xv, yv = np.meshgrid(x, y)
    initial_pos = np.column_stack((xv.flatten(), yv.flatten()))[:n_nodes].astype(np.float32)

    components = find_connected_components(n_nodes, edges)
    components.sort(key=len, reverse=True)
    
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
    print(f"  > SGLD solver simulating {len(large_comps)} large components individually.")
    print(f"  > Grouped {len(small_comps)} small components into {len(batches)} parallel batches.")
    
    final_pos = np.copy(initial_pos)
    
    node_to_comp_idx = {}
    for c_idx, comp in enumerate(active_comps):
        for node in comp:
            node_to_comp_idx[node] = c_idx
            
    comp_edges = {c_idx: [] for c_idx in range(len(active_comps))}
    comp_scores = {c_idx: [] for c_idx in range(len(active_comps))}
    
    for i, (u, v) in enumerate(edges):
        if u in node_to_comp_idx: 
            c_idx = node_to_comp_idx[u]
            comp_edges[c_idx].append((u, v))
            comp_scores[c_idx].append(edge_scores[i])

    device_rankings = Layout_Hardware.manual_layout_rankings(
        jobs, device_selection
    )
    if device_rankings is None:
        representative_indices = Layout_Hardware.representative_job_indices(
            jobs,
            node_to_comp_idx,
            comp_edges,
            params,
            engine="monte_carlo",
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
        device_rankings = {}
        for size_class, prepared in representative_batches.items():
            benchmark_params = params.copy()
            minimum_k = benchmark_params.get('SGLD_MIN_K', 20)
            fraction_k = benchmark_params.get('SGLD_K_PERCENT', 0.01)
            benchmark_params['SGLD_K'] = max(
                minimum_k, int(fraction_k * prepared.node_count)
            )
            device_rankings[size_class] = Layout_Hardware.benchmark_layout_devices(
                prepared,
                benchmark_params,
                selection=device_selection,
                size_class=size_class,
                engine_label="Monte Carlo",
                cpu_simulation_class=SSNSimulationCPU,
                gpu_simulation_class=gpu_simulation_class,
            )

    # Simulate jobs sequentially
    for job_idx, batch_comps in enumerate(jobs):
        prepared_batch = Layout_Hardware.prepare_layout_batch(
            batch_comps,
            node_to_comp_idx,
            comp_edges,
            comp_scores,
            params,
        )
        n_batch_nodes = prepared_batch.node_count
        is_large_job = prepared_batch.is_large_job
        batch_global_nodes = prepared_batch.global_nodes
        batch_edges_list = prepared_batch.edges
        batch_scores_list = prepared_batch.scores
        batch_pos = prepared_batch.positions
        batch_comp_labels = prepared_batch.component_labels
        batch_box_limits = prepared_batch.box_limits
        size_class = Layout_Hardware.layout_size_class(n_batch_nodes)
        ranked_plans = device_rankings[size_class]

        # Calculate dynamic K based on batch size: max(SGLD_MIN_K, int(SGLD_K_PERCENT * N))
        batch_params = params.copy()
        min_k = batch_params.get('SGLD_MIN_K', 20)
        pct_k = batch_params.get('SGLD_K_PERCENT', 0.01)
        batch_params['SGLD_K'] = max(min_k, int(pct_k * n_batch_nodes))

        cutoffs = [batch_params.get('SIMILARITY_THRESHOLD', 0.0)]
        if is_large_job and batch_params.get('ENABLE_PROGRESSIVE_SIMULATION', True) and n_batch_nodes > 2000 and len(batch_scores_list) > 10:
            sorted_local = np.sort(batch_scores_list)[::-1] 
            n_edges = len(sorted_local)
            fractions = [0.2, 0.4, 0.6, 0.8]
            indices = [max(0, min(int(n_edges * f) - 1, n_edges - 1)) for f in fractions]
            raw_cutoffs = [sorted_local[i] for i in indices]
            
            cutoffs = []
            for c in raw_cutoffs:
                if not cutoffs or c < cutoffs[-1]:
                    cutoffs.append(c)
                    
            if not cutoffs or cutoffs[-1] > batch_params.get('SIMILARITY_THRESHOLD', 0.0):
                cutoffs.append(batch_params.get('SIMILARITY_THRESHOLD', 0.0))
            else:
                cutoffs[-1] = batch_params.get('SIMILARITY_THRESHOLD', 0.0)
                
            print(f"  > Progressive SGLD initialized with {len(cutoffs)} stages.")
        
        previous_active = np.zeros(n_batch_nodes, dtype=np.bool_)

        for stage, cutoff in enumerate(cutoffs):
            stage_edges = [
                edge for edge, score in zip(batch_edges_list, batch_scores_list)
                if score >= cutoff
            ]
            stage_scores = [
                score for score in batch_scores_list
                if score >= cutoff
            ]
            stage_active_mask = _prepare_progressive_stage(
                batch_pos,
                stage_edges,
                stage_scores,
                previous_active,
            )
            previous_active = stage_active_mask.copy()

            if len(stage_edges) > 0:
                local_edges = np.array(stage_edges, dtype=np.int32)
            else:
                local_edges = np.zeros((0, 2), dtype=np.int32)
                
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
                        batch_params,
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
                
        final_pos[batch_global_nodes] = batch_pos
            
    # Pack independent components into a grid
    final_pos, final_box_limit = pack_components_to_grid(
        final_pos, edges, n_nodes, 
        params.get('PACKING_GRID_SIZE', 200.0), 
        params.get('PACKING_PADDING', 50.0),
        params.get('PACKING_GEOMETRY', 'Square')
    )
    return final_pos, final_box_limit


def _finite_parameter(params, name, default, *, minimum=0.0, positive=False):
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number.")
    if positive and numeric <= 0.0:
        raise ValueError(f"{name} must be greater than 0.")
    if not positive and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return numeric


def _integer_parameter(params, name, default, *, minimum, maximum):
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    numeric = int(value)
    if numeric < minimum or numeric > maximum:
        raise ValueError(
            f"{name} must be an integer from {minimum} to {maximum}."
        )
    return numeric


def _validate_energy_monte_carlo_params(params):
    if not isinstance(params, dict):
        raise ValueError("Monte Carlo params must be a dictionary.")
    selection = Hardware_Utils.normalize_device_selection(
        params.get('LAYOUT_DEVICE_SELECTION', 'auto')
    )
    if selection not in {'auto', 'cpu'}:
        raise ValueError(
            "Monte Carlo (Style) currently requires CPU execution; select "
            "Auto Benchmark or CPU."
        )
    _finite_parameter(params, 'SPRING_K', 0.1)
    _finite_parameter(params, 'COULOMB_K', 50.0)
    _finite_parameter(params, 'COULOMB_CUTOFF', 30.0)
    _finite_parameter(params, 'MAX_FORCE_LIMIT', 20.0)
    _finite_parameter(params, 'BOX_SCALE', 1.0, positive=True)
    _finite_parameter(params, 'PACKING_GRID_SIZE', 200.0, positive=True)
    _finite_parameter(params, 'PACKING_PADDING', 50.0)
    total_repulsion_limit = _finite_parameter(
        params, 'MAX_TOTAL_REPULSION_FORCE', 0.0
    )
    if total_repulsion_limit != 0.0:
        raise ValueError(
            "MAX_TOTAL_REPULSION_FORCE must be 0 for energy Monte Carlo "
            "because accumulated-force clipping has no scalar pair energy."
        )
    progressive = params.get('ENABLE_PROGRESSIVE_SIMULATION', False)
    if not isinstance(progressive, (bool, np.bool_)):
        raise ValueError("ENABLE_PROGRESSIVE_SIMULATION must be a boolean.")
    if progressive:
        raise ValueError(
            "Progressive simulation is unavailable for energy Monte Carlo "
            "because it changes the component objective during optimization."
        )
    if params.get('PACKING_GEOMETRY', 'Square') not in {'Square', 'Circle'}:
        raise ValueError("PACKING_GEOMETRY must be Square or Circle.")

    _integer_parameter(
        params, 'MC_SWEEPS', MC_SWEEPS, minimum=1, maximum=1_000_000
    )
    _integer_parameter(
        params,
        'MC_QUENCH_SWEEPS',
        MC_QUENCH_SWEEPS,
        minimum=0,
        maximum=1_000_000,
    )
    teleport_probability = params.get(
        'MC_TELEPORT_PROBABILITY', MC_TELEPORT_PROBABILITY
    )
    random_seed = params.get('MC_RANDOM_SEED', MC_RANDOM_SEED)
    if isinstance(teleport_probability, bool) or not isinstance(
        teleport_probability, (int, float, np.integer, np.floating)
    ):
        raise ValueError("MC_TELEPORT_PROBABILITY must be between 0 and 1.")
    teleport_probability = float(teleport_probability)
    if not math.isfinite(teleport_probability) or not (
        0.0 <= teleport_probability <= 1.0
    ):
        raise ValueError("MC_TELEPORT_PROBABILITY must be between 0 and 1.")
    if random_seed is not None and (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, (int, np.integer))
        or int(random_seed) < 0
    ):
        raise ValueError("MC_RANDOM_SEED must be a non-negative integer or None.")


def _validate_layout_inputs(connectivity, n_nodes):
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)):
        raise ValueError("n_nodes must be a non-negative integer.")
    node_count = int(n_nodes)
    if node_count < 0:
        raise ValueError("n_nodes must be a non-negative integer.")

    try:
        array = np.asarray(connectivity)
    except (TypeError, ValueError) as error:
        raise ValueError("connectivity must be a numeric M x 3 array.") from error
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("connectivity must be a two-dimensional M x 3 array.")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError("connectivity must contain finite real numeric values.")
    try:
        finite = np.isfinite(array).all()
    except TypeError as error:
        raise ValueError("connectivity must contain finite numeric values.") from error
    if not finite:
        raise ValueError("connectivity must contain finite numeric values.")

    endpoints = array[:, :2]
    if endpoints.size and not np.equal(endpoints, np.floor(endpoints)).all():
        raise ValueError("connectivity endpoints must be integer node indices.")
    if endpoints.size and (
        np.min(endpoints) < 0 or np.max(endpoints) >= node_count
    ):
        raise ValueError("connectivity contains a node index outside n_nodes.")
    return array, node_count


def calculate_layout(connectivity, n_nodes, params):
    """Minimize topology-only energy with uniform springs on retained edges."""
    _validate_energy_monte_carlo_params(params)
    connectivity, n_nodes = _validate_layout_inputs(connectivity, n_nodes)
    edges = np.asarray(connectivity[:, :2], dtype=np.int32)
    edge_scores = np.asarray(connectivity[:, 2], dtype=np.float64)

    print("Computing initial node positions using Laplacian Spectral / Grid layouts...")
    side = int(np.ceil(np.sqrt(n_nodes))) if n_nodes else 0
    base_box = np.sqrt(n_nodes) * 2.5 + 5.0 if n_nodes else 5.0
    initial_box_limit = base_box * params.get('BOX_SCALE', 1.0)
    if n_nodes:
        axis = np.linspace(
            -initial_box_limit * 0.5,
            initial_box_limit * 0.5,
            side,
        )
        grid_x, grid_y = np.meshgrid(axis, axis)
        final_pos = np.column_stack(
            (grid_x.flatten(), grid_y.flatten())
        )[:n_nodes].astype(np.float32)
    else:
        final_pos = np.zeros((0, 2), dtype=np.float32)

    components = find_connected_components(n_nodes, edges)
    components.sort(key=len, reverse=True)
    active_components = [component for component in components if len(component) > 1]
    print(f"Found {len(active_components)} active components.")
    print("  > Energy Monte Carlo uses exact CPU component objectives.")

    node_to_component = {}
    for component_index, component in enumerate(active_components):
        for node in component:
            node_to_component[node] = component_index
    component_edges = {
        component_index: [] for component_index in range(len(active_components))
    }
    component_scores = {
        component_index: [] for component_index in range(len(active_components))
    }
    for edge_index, (source, target) in enumerate(edges):
        component_index = node_to_component.get(int(source))
        if component_index is not None:
            component_edges[component_index].append((int(source), int(target)))
            component_scores[component_index].append(float(edge_scores[edge_index]))

    random_seed = params.get('MC_RANDOM_SEED', MC_RANDOM_SEED)
    seed_sequence = np.random.SeedSequence(
        None if random_seed is None else int(random_seed)
    )
    component_seeds = seed_sequence.spawn(len(active_components))

    for component_index, (component, component_seed) in enumerate(
        zip(active_components, component_seeds)
    ):
        rng = np.random.default_rng(component_seed)
        prepared = Layout_Hardware.prepare_layout_batch(
            [component],
            node_to_component,
            component_edges,
            component_scores,
            params,
            add_noise=True,
            verbose=True,
            rng=rng,
        )
        optimizer = ComponentEnergyMonteCarlo(
            prepared.positions,
            prepared.edges,
            prepared.box_limits,
            params,
            rng,
        )
        print(
            f"\nOptimizing component {component_index + 1}/"
            f"{len(active_components)} ({len(component)} nodes)..."
        )
        component_positions, component_energy = optimizer.optimize()
        print(
            f"  > Best component energy: {component_energy:.8g}; "
            f"maximum reconciliation drift: {optimizer.max_energy_drift:.3g}"
        )
        final_pos[prepared.global_nodes] = component_positions

    if n_nodes == 0:
        return final_pos, 0.0
    final_pos, final_box_limit = pack_components_to_grid(
        final_pos,
        edges,
        n_nodes,
        params.get('PACKING_GRID_SIZE', 200.0),
        params.get('PACKING_PADDING', 50.0),
        params.get('PACKING_GEOMETRY', 'Square'),
    )
    return final_pos, final_box_limit
