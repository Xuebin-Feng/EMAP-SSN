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

try:
    import torch
    try:
        from utilities import Hardware_Utils, Layout_Hardware
    except ImportError:
        import Hardware_Utils, Layout_Hardware
    HAS_TORCH = True
except Exception as e:
    import traceback
    print("Warning: PyTorch or Hardware_Utils could not be imported. GPU acceleration will be disabled.")
    print(f"Detail: {e}")
    traceback.print_exc()
    HAS_TORCH = False

# --- 1. Physics Kernels ---

def _get_physics_kernel():
    def _run_physics_kernel(pos, vel, springs, comp_labels, active_mask, active_nodes, box_limits, dt, damping, k_spr, k_coul, max_f, max_total_repulsion, cutoff_dist):
        n_balls = pos.shape[0]
        acc = np.zeros_like(pos)
        repulsion = np.zeros_like(pos)
        
        # Calculate squared cutoff for efficient distance comparison
        cutoff_sq = cutoff_dist * cutoff_dist 
        taper_start = cutoff_dist * 0.8
        taper_width = max(cutoff_dist * 0.2, 1e-9)
        
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
        for active_i in range(active_nodes.shape[0]):
            i = active_nodes[active_i]
            for active_j in range(active_i + 1, active_nodes.shape[0]):
                j = active_nodes[active_j]
                if comp_labels[i] != comp_labels[j]:
                    continue

                dx, dy = pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]
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

                repulsion[i, 0] += f*(dx/dist); repulsion[i, 1] += f*(dy/dist)
                repulsion[j, 0] -= f*(dx/dist); repulsion[j, 1] -= f*(dy/dist)

        # MAX_FORCE_LIMIT caps each pair. This second cap limits the norm of the
        # accumulated repulsive force on a node before it is combined with springs.
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
                    
        # --- INTEGRATION (Euler) ---
        rmsd = 0.0
        n_active = active_nodes.shape[0]
        for i in range(n_balls):
            if not active_mask[i]:
                continue
            box_limit = box_limits[i]
            acc[i] -= damping * vel[i]
            vel[i] += acc[i] * dt
            old_p = pos[i].copy()
            pos[i] += vel[i] * dt
            
            if pos[i,0] > box_limit: pos[i,0]=box_limit; vel[i,0]*=-0.5
            elif pos[i,0] < -box_limit: pos[i,0]=-box_limit; vel[i,0]*=-0.5
            if pos[i,1] > box_limit: pos[i,1]=box_limit; vel[i,1]*=-0.5
            elif pos[i,1] < -box_limit: pos[i,1]=-box_limit; vel[i,1]*=-0.5
            
            diff = pos[i] - old_p
            rmsd += diff[0]**2 + diff[1]**2
            
        if n_active == 0:
            return 0.0
        return math.sqrt(rmsd / n_active)
        
    if NUMBA_AVAILABLE:
        return jit(nopython=True, fastmath=True)(_run_physics_kernel)
    return _run_physics_kernel

run_physics_kernel = _get_physics_kernel()

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

    def step(self, current_step):
        return run_physics_kernel(
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
            out_of_bounds_x = (self.pos[:, 0].abs() > self.box_limits) & self.active_mask
            out_of_bounds_y = (self.pos[:, 1].abs() > self.box_limits) & self.active_mask
            
            # Reverse and dampen velocity for nodes hitting the walls
            self.vel[out_of_bounds_x, 0] *= -0.5
            self.vel[out_of_bounds_y, 1] *= -0.5
            
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
    """Maps each node to its connected component ID for isolated physics."""
    components = find_connected_components(n_nodes, edges)
    labels = np.zeros(n_nodes, dtype=np.int32)
    for c_id, comp in enumerate(components):
        for node in comp:
            labels[node] = c_id
    return labels

def pack_components_to_grid(pos, edges, n_nodes, grid_size, padding, packing_geometry="Square"):
    """Packs independent network components into a strict master grid layout."""
    print("Packing independent components using macro-grid boolean packing...")
    components = find_connected_components(n_nodes, edges)
    
    if not components:
        return pos, 100.0

    # --- 1. Map edges to components for fast lookup ---
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
        
        # Use a top-left origin approach so Y goes downwards into grid rows
        min_x = np.min(comp_pos[:, 0])
        max_y = np.max(comp_pos[:, 1]) 
        
        shifted_pos = comp_pos - [min_x, max_y] # X is >= 0, Y is <= 0
        
        global_to_local = {g: l for l, g in enumerate(comp)}
        points = list(shifted_pos)
        
        # Rasterize edges so we don't accidentally place a dot on a connecting line
        for u, v in comp_edges[c_id]:
            p1 = shifted_pos[global_to_local[u]]
            p2 = shifted_pos[global_to_local[v]]
            dist = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            
            # Sample points along the edge line
            steps = int(dist / (grid_size / 4)) + 1
            for i in range(1, steps):
                t = i / steps
                px = p1[0] + t * (p2[0] - p1[0])
                py = p1[1] + t * (p2[1] - p1[1])
                points.append([px, py])
                
        # --- Determine Grid Footprint ---
        max_c, max_r = 0, 0
        pad = padding / 2.0
        
        # First pass: find the maximum grid cells required
        for px, py in points:
            pos_y = -py # Invert Y so positive goes down into rows
            c_max = int((px + pad) / grid_size)
            r_max = int((pos_y + pad) / grid_size)
            max_c = max(max_c, c_max)
            max_r = max(max_r, r_max)
            
        cols = max_c + 1
        rows = max_r + 1
        mask = np.zeros((rows, cols), dtype=bool)
        
        # Second pass: mark cells as occupied
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
        
    # --- 2. Sort components: Largest area first, then tie-break by node count ---
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
            
    # --- 5. Center the final visualization ---
    global_min = np.min(new_pos, axis=0)
    global_max = np.max(new_pos, axis=0)
    center = (global_max + global_min) / 2.0
    new_pos -= center
    
    new_box_limit = max(global_max[0] - global_min[0], global_max[1] - global_min[1]) / 2.0 * 1.1
    
    print(f"Packed {len(components)} objects into a uniform grid. Ready for display.")
    return new_pos, new_box_limit

# --- 3. Main Layout Algorithm ---

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

    # Prefer neighbors that were already relaxed in the preceding stage.
    for (u, v), score in zip(stage_edges, stage_scores):
        weight = max(float(score), 1e-9)
        if newly_active[u] and previous_active[v]:
            weighted_sum[u] += reference_pos[v] * weight
            weight_sum[u] += weight
        if newly_active[v] and previous_active[u]:
            weighted_sum[v] += reference_pos[u] * weight
            weight_sum[v] += weight

    # If a newly activated group has no older anchor, retain its spectral/grid
    # initialization rather than forcing several new nodes onto one coordinate.
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
                        if percentage_drop < percentage_threshold:
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


def calculate_layout(connectivity, n_nodes, params):
    """
    Main layout generation pipeline.
    
    connectivity: N x 3 NumPy array representing [Source_Index, Target_Index, Score]
    n_nodes: Total number of nodes in the network
    params: Dictionary containing physics and execution parameters
    
    Returns:
        pos (np.ndarray): Final X/Y coordinates
        box_limit (float): Boundary box size
    """
    device_selection = params.get('LAYOUT_DEVICE_SELECTION', 'auto')
    
    edges = connectivity[:, :2].astype(np.int32)
    edge_scores = connectivity[:, 2]
    
    # Initialize basic grid positioning to start
    side = int(np.ceil(np.sqrt(n_nodes)))
    base_box = np.sqrt(n_nodes) * 2.5 + 5.0
    initial_box_limit = base_box * params.get('BOX_SCALE', 1.0)
    x = np.linspace(-initial_box_limit*0.5, initial_box_limit*0.5, side)
    y = np.linspace(-initial_box_limit*0.5, initial_box_limit*0.5, side)
    xv, yv = np.meshgrid(x, y)
    initial_pos = np.column_stack((xv.flatten(), yv.flatten()))[:n_nodes].astype(np.float32)

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
    
    # Pre-map edges and scores to components for O(1) extraction
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
    
    # 3. Simulate jobs sequentially
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

        if is_large_job:
             print(f"\nSimulating Large Component {job_idx+1}/{len(jobs)} ({n_batch_nodes} nodes)...")
        else:
             print(f"\nSimulating Batch {job_idx+1}/{len(jobs)} ({len(batch_comps)} components, {n_batch_nodes} nodes)...")

        cutoffs = [params.get('SIMILARITY_THRESHOLD', 0.0)]
        if is_large_job and params.get('ENABLE_PROGRESSIVE_SIMULATION', True) and n_batch_nodes > 2000 and len(batch_scores_list) > 10:
            sorted_local = np.sort(batch_scores_list)[::-1] 
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
            if len(cutoffs) > 1:
                stage_edge_count = sum(1 for s in batch_scores_list if s >= cutoff)
                print(f"  > Stage {stage+1}/{len(cutoffs)}: Cutoff = {cutoff:.3f} | Active Edges: {stage_edge_count}")

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
                        params,
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
    final_pos, final_box_limit = pack_components_to_grid(
        final_pos, edges, n_nodes, 
        params.get('PACKING_GRID_SIZE', 200.0), 
        params.get('PACKING_PADDING', 50.0),
        params.get('PACKING_GEOMETRY', 'Square')
    )
    
    return final_pos, final_box_limit
