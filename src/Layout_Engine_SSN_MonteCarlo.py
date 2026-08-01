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
        from utilities import Hardware_Utils
    except ImportError:
        import Hardware_Utils
    HAS_TORCH = True
except Exception as e:
    import traceback
    print("Warning: PyTorch or Hardware_Utils could not be imported. GPU acceleration will be disabled.")
    print(f"Detail: {e}")
    traceback.print_exc()
    HAS_TORCH = False

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
        def __init__(self, pos, springs, comp_labels, box_limit, params, active_mask=None):
            self.device = Hardware_Utils.get_optimal_device()
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


def calculate_layout(connectivity, n_nodes, params):
    """
    Main layout generation pipeline using Monte Carlo SGLD.
    """
    # Determine execution device
    use_gpu = False
    device_name = "CPU"
    if HAS_TORCH:
        device = Hardware_Utils.get_optimal_device()
        if device.type != "cpu":
            use_gpu = True
            device_name = f"GPU ({device})"
            
    print(f"Running layout calculation on {device_name}")
    
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
            
    # Simulate jobs sequentially
    for job_idx, batch_comps in enumerate(jobs):
        n_batch_nodes = sum(len(c) for c in batch_comps)
        is_large_job = len(batch_comps) == 1 and n_batch_nodes >= 500

        batch_global_nodes = []
        for c in batch_comps:
            batch_global_nodes.extend(c)

        global_to_batch = {g_id: l_id for l_id, g_id in enumerate(batch_global_nodes)}

        batch_edges_list = []
        batch_scores_list = []
        batch_pos_list = []
        batch_comp_labels_list = []
        batch_box_limits_list = []

        for c_idx_in_batch, c in enumerate(batch_comps):
            n_comp_nodes = len(c)
            c_idx = node_to_comp_idx[c[0]]

            c_edges = comp_edges[c_idx]
            c_scores = comp_scores[c_idx]

            comp_global_to_local = {g: l for l, g in enumerate(c)}
            c_local_edges = [(comp_global_to_local[u], comp_global_to_local[v]) for u, v in c_edges]

            comp_box_limit = (np.sqrt(n_comp_nodes) * 2.5 + 5.0) * params.get('BOX_SCALE', 1.0)
            local_pos = None
            spectral_success = False

            if n_comp_nodes >= 4:
                if n_comp_nodes >= 50:
                    print(f"  > Calculating Spectral Layout for sub-component ({n_comp_nodes} nodes)...")
                try:
                    import scipy.sparse as sp
                    from scipy.sparse.csgraph import laplacian
                    from scipy.sparse.linalg import eigsh

                    row = [e[0] for e in c_local_edges] + [e[1] for e in c_local_edges]
                    col = [e[1] for e in c_local_edges] + [e[0] for e in c_local_edges]
                    data = c_scores + c_scores
                    adj = sp.coo_matrix((data, (row, col)), shape=(n_comp_nodes, n_comp_nodes))

                    L = laplacian(adj, normed=True)
                    vals, vecs = eigsh(L, k=3, which='SM', tol=1e-3)

                    x_coords = vecs[:, 1]
                    y_coords = vecs[:, 2]

                    x_norm = (x_coords - np.min(x_coords)) / (np.ptp(x_coords) + 1e-9)
                    y_norm = (y_coords - np.min(y_coords)) / (np.ptp(y_coords) + 1e-9)

                    x_scaled = (x_norm - 0.5) * comp_box_limit * 0.8
                    y_scaled = (y_norm - 0.5) * comp_box_limit * 0.8

                    local_pos = np.column_stack((x_scaled, y_scaled)).astype(np.float32)
                    spectral_success = True
                except Exception as e:
                    if n_comp_nodes >= 50:
                        print(f"  > Spectral solver failed: {e}. Falling back to grid layout.")

            if not spectral_success:
                side_comp = int(np.ceil(np.sqrt(n_comp_nodes)))
                x_c = np.linspace(-comp_box_limit * 0.5, comp_box_limit * 0.5, side_comp)
                y_c = np.linspace(-comp_box_limit * 0.5, comp_box_limit * 0.5, side_comp)
                xv_c, yv_c = np.meshgrid(x_c, y_c)
                local_pos = np.column_stack((xv_c.flatten(), yv_c.flatten()))[:n_comp_nodes].astype(np.float32)

            # Components share one vectorized batch while retaining independent
            # local coordinate systems centered at the origin.
            local_min = np.min(local_pos, axis=0)
            local_max = np.max(local_pos, axis=0)
            local_pos -= (local_min + local_max) / 2.0

            batch_pos_list.append(local_pos)
            batch_comp_labels_list.append(
                np.full(n_comp_nodes, c_idx_in_batch, dtype=np.int32)
            )
            batch_box_limits_list.append(
                np.full(n_comp_nodes, comp_box_limit, dtype=np.float32)
            )

            for (u, v), score in zip(c_edges, c_scores):
                batch_edges_list.append((global_to_batch[u], global_to_batch[v]))
                batch_scores_list.append(score)

        batch_pos = np.vstack(batch_pos_list).astype(np.float32)
        batch_comp_labels = np.concatenate(batch_comp_labels_list)
        batch_box_limits = np.concatenate(batch_box_limits_list)
        batch_pos += np.random.normal(0, 0.1, batch_pos.shape).astype(np.float32)

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
                
            if use_gpu:
                sim = SSNSimulationGPU(
                    batch_pos, local_edges, batch_comp_labels,
                    batch_box_limits, batch_params,
                    active_mask=stage_active_mask
                )
            else:
                sim = SSNSimulationCPU(
                    batch_pos, local_edges, batch_comp_labels,
                    batch_box_limits, batch_params,
                    active_mask=stage_active_mask
                )
                
            rmsd_window = batch_params.get('RMSD_WINDOW', 50)
            max_steps = batch_params.get('MAX_STEPS', 2000)
            rmsd_buffer = deque(maxlen=rmsd_window)
            avg_history = []
            
            for step in range(max_steps):
                rmsd = sim.step(step)
                rmsd_buffer.append(rmsd)
                avg_rmsd = np.mean(rmsd_buffer)
                
                 # Progress reporting every 500 steps
                if step > 0 and step % 500 == 0:
                    print(f"    - Step {step:04d}/{max_steps} | RMSD: {avg_rmsd:.5f}")
                
                if len(rmsd_buffer) == rmsd_window:
                    avg_history.append(avg_rmsd)
                    
                    progress = step / float(max_steps)
                    if avg_rmsd < batch_params.get('RMSD_THRESHOLD', 0.005) and progress > 0.8:
                        print(f"    - Converged at Step {step} (RMSD: {avg_rmsd:.5f})")
                        break
                        
                    pct_threshold = batch_params.get('PERCENTAGE_DROP_THRESHOLD', 0.0)
                    minimum_observation_steps = max_steps / 4.0
                    trend_window = 10
                    
                    if pct_threshold > 0.0 and len(avg_history) >= (rmsd_window + trend_window) and step > minimum_observation_steps:
                        current_trend = np.mean(avg_history[-trend_window:])
                        old_trend = np.mean(avg_history[-(rmsd_window + trend_window):-rmsd_window])
                        
                        if old_trend > 0:
                            pct_drop = ((old_trend - current_trend) / old_trend) * 100.0
                            if pct_drop < pct_threshold and progress > 0.8:
                                print(f"    - Plateau Reached at Step {step} (Drop: {pct_drop:.3f}% < {pct_threshold}%)")
                                break
                    
            batch_pos = sim.get_pos()
            del sim
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        final_pos[batch_global_nodes] = batch_pos
            
    # Pack independent components into a grid
    final_pos, final_box_limit = pack_components_to_grid(
        final_pos, edges, n_nodes, 
        params.get('PACKING_GRID_SIZE', 200.0), 
        params.get('PACKING_PADDING', 50.0),
        params.get('PACKING_GEOMETRY', 'Square')
    )
    return final_pos, final_box_limit
