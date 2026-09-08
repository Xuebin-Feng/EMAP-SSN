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
import pandas as pd
import warnings

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def calculate_layout(connectivity, n_nodes, params):
    """
    Alternative layout generation pipeline using UMAP with native k-NN precomputed tuples.
    
    connectivity: N x 3 NumPy array representing [Source_Index, Target_Index, Score]
    n_nodes: Total number of nodes in the network
    params: Dictionary containing execution parameters
    
    Returns:
        pos (np.ndarray): Final X/Y coordinates
        box_limit (float): Boundary box size
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP library is not installed. Please install 'umap-learn' to use the UMAP layout engine.")
        
    print(f"Running UMAP global layout on {n_nodes} nodes...")
    
    target_box = (np.sqrt(n_nodes) * 2.5 + 5.0)
    final_box_limit = target_box * params.get('BOX_SCALE', 1.0)

    if connectivity.shape[0] == 0:
        # Edge case: No edges at all
        print("Warning: Network has no edges. Generating random layout.")
        pos = np.random.uniform(-final_box_limit / 2.0, final_box_limit / 2.0, (n_nodes, 2)).astype(np.float32)
        return pos, final_box_limit

    # Extract connectivity data
    sources = connectivity[:, 0].astype(np.int32)
    targets = connectivity[:, 1].astype(np.int32)
    scores = connectivity[:, 2].astype(np.float32)

    # --- 1. Score-to-Distance Conversion ---
    # UMAP expects distances >= 0. High similarities become distances near 0.
    # Maximum similarity (identical sequences) has exact distance 0.0 ("no difference").
    max_score = np.max(scores)
    distances = max_score - scores

    # --- 2. Build Explicit k-NN Tuples (Method 2) ---
    # Decouples "no difference" (distance 0.0) from "no edge" (index -1, distance inf).
    # User K counts other nodes; UMAP also requires self in column zero.
    requested_neighbors = params.get('UMAP_NEIGHBORS', 15)
    real_neighbors = min(requested_neighbors, n_nodes - 1)
    n_neighbors = real_neighbors + 1
        
    min_dist = params.get('UMAP_MIN_DIST', 0.1)

    all_u = np.concatenate([sources, targets])
    all_v = np.concatenate([targets, sources])
    all_d = np.concatenate([distances, distances])

    df = pd.DataFrame({'u': all_u, 'v': all_v, 'd': all_d})
    # Exclude self-loops and duplicate edges
    df = df[df['u'] != df['v']]
    df = df.sort_values(['u', 'd'], ascending=[True, True])
    df = df.drop_duplicates(subset=['u', 'v'])
    top = df.groupby('u').head(real_neighbors)

    knn_indices = np.full((n_nodes, n_neighbors), -1, dtype=np.int32)
    knn_dists = np.full((n_nodes, n_neighbors), np.inf, dtype=np.float32)
    knn_indices[:, 0] = np.arange(n_nodes)
    knn_dists[:, 0] = 0.0

    top = top.assign(rank=top.groupby('u').cumcount() + 1)
    knn_indices[top['u'].values, top['rank'].values] = top['v'].values
    knn_dists[top['u'].values, top['rank'].values] = top['d'].values

    # --- 3. Run UMAP with Native precomputed_knn ---
    print(f"  > Initializing UMAP (K={requested_neighbors} other nodes, "
          f"up to {real_neighbors} available + self, min_dist={min_dist})")
    warnings.filterwarnings('ignore', category=UserWarning, module='umap')
    X_dummy = np.zeros((n_nodes, 1), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            precomputed_knn=(knn_indices, knn_dists, None),
            init='random' if n_nodes <= 3 else 'spectral',
            random_state=42 # fixed seed for reproducible deterministic layouts
        )
        final_pos = reducer.fit_transform(X_dummy).astype(np.float32)

    # --- 4. Handle Disconnected Vertices (NaN coordinates) ---
    nan_mask = ~np.isfinite(final_pos).all(axis=1)
    if np.any(nan_mask):
        rng = np.random.RandomState(42)
        final_pos[nan_mask] = rng.uniform(-0.5, 0.5, (np.sum(nan_mask), 2)).astype(np.float32)

    # --- 5. Scale and Center Output ---
    connected_mask = ~nan_mask
    if np.any(connected_mask):
        global_min = np.min(final_pos[connected_mask], axis=0)
        global_max = np.max(final_pos[connected_mask], axis=0)
        center = (global_max + global_min) / 2.0
        final_pos -= center
        ptp = np.ptp(final_pos[connected_mask], axis=0)
        umap_max_spread = max(ptp[0], ptp[1]) + 1e-9  # avoid division by zero
    else:
        umap_max_spread = 1.0

    scale_factor = (final_box_limit * 1.6) / umap_max_spread 
    final_pos *= scale_factor

    print("UMAP layout calculation complete.")
    return final_pos, final_box_limit
