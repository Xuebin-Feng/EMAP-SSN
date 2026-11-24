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

import contextlib
import functools
import inspect
import sys
import threading
import warnings

import numpy as np

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

try:
    from utilities import Numba_Threads
except ImportError:
    import Numba_Threads

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _build_symmetric_csr(sources, targets, distances, n_nodes):
        """Build undirected CSR (indptr, adj_v, adj_d) in O(E + N), skipping self-loops."""
        n_edges = sources.shape[0]
        indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        for e in range(n_edges):
            u = sources[e]
            v = targets[e]
            if u < 0 or u >= n_nodes or v < 0 or v >= n_nodes:
                raise ValueError("Edge endpoint index out of bounds.")
            if u != v:
                indptr[u + 1] += 1
                indptr[v + 1] += 1
        for i in range(n_nodes):
            indptr[i + 1] += indptr[i]
        total = indptr[n_nodes]
        adj_v = np.empty(total, dtype=np.int32)
        adj_d = np.empty(total, dtype=np.float32)
        fill = indptr[:-1].copy()
        for e in range(n_edges):
            u = sources[e]
            v = targets[e]
            if u != v:
                d = distances[e]
                pu = fill[u]
                fill[u] = pu + 1
                adj_v[pu] = v
                adj_d[pu] = d
                pv = fill[v]
                fill[v] = pv + 1
                adj_v[pv] = u
                adj_d[pv] = d
        return indptr, adj_v, adj_d

    @njit(parallel=True, cache=True)
    def _csr_topk_dedup(indptr, adj_v, adj_d, n_nodes, real_neighbors):
        """Parallel per-node deduplicated top-K nearest neighbors in ascending (d, v) order."""
        n_neighbors = real_neighbors + 1
        knn_indices = np.full((n_nodes, n_neighbors), -1, dtype=np.int32)
        knn_dists = np.full((n_nodes, n_neighbors), np.inf, dtype=np.float32)

        for u in prange(n_nodes):
            knn_indices[u, 0] = u
            knn_dists[u, 0] = np.float32(0.0)
            start = indptr[u]
            end = indptr[u + 1]
            if start == end or real_neighbors <= 0:
                continue

            if real_neighbors <= 64:
                # Single-pass O(deg) bounded insertion list (no sorting of whole row!).
                best_v = np.empty(real_neighbors, dtype=np.int32)
                best_d = np.empty(real_neighbors, dtype=np.float32)
                taken = 0
                for p in range(start, end):
                    v = adj_v[p]
                    d = adj_d[p]
                    if not np.isfinite(d):
                        continue
                    if taken == real_neighbors:
                        wd = best_d[taken - 1]
                        if d > wd or (d == wd and v >= best_v[taken - 1]):
                            continue
                    # Check if v is already in best_v[:taken] (handles duplicate edges).
                    dup_pos = -1
                    for j in range(taken):
                        if best_v[j] == v:
                            dup_pos = j
                            break
                    if dup_pos != -1:
                        if d < best_d[dup_pos]:
                            best_d[dup_pos] = d
                            pos = dup_pos
                            while pos > 0 and (
                                best_d[pos - 1] > d
                                or (best_d[pos - 1] == d and best_v[pos - 1] > v)
                            ):
                                best_d[pos] = best_d[pos - 1]
                                best_v[pos] = best_v[pos - 1]
                                pos -= 1
                            best_d[pos] = d
                            best_v[pos] = v
                    else:
                        if taken < real_neighbors:
                            pos = taken
                            taken += 1
                        else:
                            pos = taken - 1
                        while pos > 0 and (
                            best_d[pos - 1] > d
                            or (best_d[pos - 1] == d and best_v[pos - 1] > v)
                        ):
                            best_d[pos] = best_d[pos - 1]
                            best_v[pos] = best_v[pos - 1]
                            pos -= 1
                        best_d[pos] = d
                        best_v[pos] = v
                for j in range(taken):
                    knn_indices[u, j + 1] = best_v[j]
                    knn_dists[u, j + 1] = best_d[j]
            else:
                # General K > 64: sort node u's row by (v, d) to deduplicate, then by (d, v).
                deg = end - start
                rv = np.empty(deg, dtype=np.int32)
                rd = np.empty(deg, dtype=np.float32)
                n_fin = 0
                for p in range(start, end):
                    d = adj_d[p]
                    if not np.isfinite(d):
                        continue
                    rv[n_fin] = adj_v[p]
                    rd[n_fin] = d
                    n_fin += 1
                if n_fin == 0:
                    continue
                # Shell sort by (v, d) to collapse duplicates to min distance.
                gap = 1
                while gap < n_fin // 3:
                    gap = gap * 3 + 1
                while gap > 0:
                    for i in range(gap, n_fin):
                        kv = rv[i]
                        kd = rd[i]
                        j = i - gap
                        while j >= 0 and (rv[j] > kv or (rv[j] == kv and rd[j] > kd)):
                            rv[j + gap] = rv[j]
                            rd[j + gap] = rd[j]
                            j -= gap
                        rv[j + gap] = kv
                        rd[j + gap] = kd
                    gap //= 3
                uniq = 0
                for i in range(n_fin):
                    if uniq == 0 or rv[i] != rv[uniq - 1]:
                        rv[uniq] = rv[i]
                        rd[uniq] = rd[i]
                        uniq += 1
                # Shell sort unique entries by (d, v).
                gap = 1
                while gap < uniq // 3:
                    gap = gap * 3 + 1
                while gap > 0:
                    for i in range(gap, uniq):
                        kv = rv[i]
                        kd = rd[i]
                        j = i - gap
                        while j >= 0 and (rd[j] > kd or (rd[j] == kd and rv[j] > kv)):
                            rv[j + gap] = rv[j]
                            rd[j + gap] = rd[j]
                            j -= gap
                        rv[j + gap] = kv
                        rd[j + gap] = kd
                    gap //= 3
                take = uniq if uniq < real_neighbors else real_neighbors
                for j in range(take):
                    knn_indices[u, j + 1] = rv[j]
                    knn_dists[u, j + 1] = rd[j]

        return knn_indices, knn_dists


def build_precomputed_knn(sources, targets, distances, n_nodes, real_neighbors):
    """Build exact deduplicated UMAP (knn_indices, knn_dists) arrays."""
    sources = np.asarray(sources)
    targets = np.asarray(targets)
    distances = np.asarray(distances)
    n_nodes = int(n_nodes)
    real_neighbors = int(real_neighbors)
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n_nodes}.")
    if sources.size != targets.size or sources.size != distances.size:
        raise ValueError(
            f"sources ({sources.size}), targets ({targets.size}), and "
            f"distances ({distances.size}) must have the same length."
        )
    if sources.size > 0:
        src_min, src_max = int(sources.min()), int(sources.max())
        dst_min, dst_max = int(targets.min()), int(targets.max())
        if src_min < 0 or src_max >= n_nodes or dst_min < 0 or dst_max >= n_nodes:
            raise ValueError(
                f"Edge endpoints must lie in [0, {n_nodes}); got "
                f"sources in [{src_min}, {src_max}], targets in [{dst_min}, {dst_max}]."
            )
    if _NUMBA_AVAILABLE:
        with Numba_Threads.limited_threads(Numba_Threads.default_thread_count()):
            indptr, adj_v, adj_d = _build_symmetric_csr(
                sources, targets, distances, n_nodes
            )
            return _csr_topk_dedup(
                indptr, adj_v, adj_d, n_nodes, real_neighbors
            )
    import pandas as pd

    n_neighbors = real_neighbors + 1
    all_u = np.concatenate([sources, targets])
    all_v = np.concatenate([targets, sources])
    all_d = np.concatenate([distances, distances])
    df = pd.DataFrame({'u': all_u, 'v': all_v, 'd': all_d})
    df = df[(df['u'] != df['v']) & np.isfinite(df['d'])]
    df = df.sort_values(['u', 'd', 'v'], ascending=[True, True, True])
    df = df.drop_duplicates(subset=['u', 'v'])
    top = df.groupby('u').head(real_neighbors)

    knn_indices = np.full((n_nodes, n_neighbors), -1, dtype=np.int32)
    knn_dists = np.full((n_nodes, n_neighbors), np.inf, dtype=np.float32)
    knn_indices[:, 0] = np.arange(n_nodes)
    knn_dists[:, 0] = 0.0

    top = top.assign(rank=top.groupby('u').cumcount() + 1)
    knn_indices[top['u'].values, top['rank'].values] = top['v'].values
    knn_dists[top['u'].values, top['rank'].values] = top['d'].values
    return knn_indices, knn_dists


# --- Seeded ARPACK ---------------------------------------------------------------------
# umap's spectral start calls scipy.sparse.linalg.eigsh for every connected component of
# the fuzzy graph (and scikit-learn's SpectralEmbedding calls it to place the components).
# SciPy 1.18's ARPACK draws a random vector whenever its Krylov space runs out, as it does
# at once on a small component of interchangeable nodes (a clique of equal scores), and
# takes it from a Generator that eigsh builds from its `rng` argument. umap never passes
# `rng`, so each such call starts from fresh operating-system entropy: the spectral start,
# and through it the whole layout, differs between runs of the same seed, with a cold or a
# warm Numba cache alike. Inside _seeded_arpack every eigsh call made by the running thread
# gets rng=<layout seed>. When ARPACK never needs a random vector the argument goes unused
# and the layout is bit-identical to an unwrapped run.
_ARPACK_SEEDS = threading.local()
_ARPACK_LOCK = threading.Lock()
_ARPACK_PATCH = {'users': 0, 'original': None, 'wrapper': None, 'holders': []}
_ARPACK_HOLDER_PREFIXES = ('sklearn.', 'umap.', 'pynndescent.')


def _seeding_eigsh(original):
    @functools.wraps(original)
    def eigsh(*args, **kwargs):
        seed = getattr(_ARPACK_SEEDS, 'value', None)
        if seed is not None and kwargs.get('rng') is None:
            kwargs['rng'] = seed
        return original(*args, **kwargs)
    return eigsh


def _holds_eigsh(module, original):
    try:
        return module is not None and getattr(module, 'eigsh', None) is original
    except Exception:  # a lazy or half-imported module that cannot answer is not a holder
        return False


def _install_seeding_eigsh():
    """Put the seeding wrapper where umap and scikit-learn look eigsh up (reference counted)."""
    import scipy.sparse.linalg as sparse_linalg
    with _ARPACK_LOCK:
        patch = _ARPACK_PATCH
        if patch['users'] == 0:
            original = sparse_linalg.eigsh
            if 'rng' in inspect.signature(original).parameters:
                wrapper = _seeding_eigsh(original)
                holders = [sparse_linalg] + [
                    module for name, module in list(sys.modules.items())
                    if name.startswith(_ARPACK_HOLDER_PREFIXES) and _holds_eigsh(module, original)
                ]
                for module in holders:
                    module.eigsh = wrapper
                patch.update(original=original, wrapper=wrapper, holders=holders)
        patch['users'] += 1


def _remove_seeding_eigsh():
    with _ARPACK_LOCK:
        patch = _ARPACK_PATCH
        patch['users'] -= 1
        if patch['users'] == 0 and patch['wrapper'] is not None:
            for module in patch['holders']:
                if getattr(module, 'eigsh', None) is patch['wrapper']:
                    module.eigsh = patch['original']
            patch.update(original=None, wrapper=None, holders=[])


@contextlib.contextmanager
def _seeded_arpack(seed):
    """Give every eigsh call this thread makes rng=seed; a seed of None leaves eigsh unseeded."""
    if seed is None:
        yield
        return
    previous = getattr(_ARPACK_SEEDS, 'value', None)
    _install_seeding_eigsh()
    _ARPACK_SEEDS.value = seed
    try:
        yield
    finally:
        _ARPACK_SEEDS.value = previous
        _remove_seeding_eigsh()


def _resolve_seed(params):
    """Return the layout seed, or None for non-reproducible behaviour.

    Kept behaviourally identical to Layout_Engine_SSN._resolve_seed. The two
    engines share no imports by design - the only contract between them is
    calculate_layout(connectivity, n_nodes, params) - so the check is
    duplicated rather than imported.
    """
    value = params.get('LAYOUT_SEED', 42)
    if value is None:
        return None
    seed = int(value)
    if seed < 0:
        raise ValueError(
            f"LAYOUT_SEED must be a non-negative integer or null, got {value!r}."
        )
    return seed


def calculate_layout(connectivity, n_nodes, params):
    """
    Alternative layout generation pipeline using UMAP with native k-NN precomputed tuples.
    
    connectivity: N x 3 NumPy array representing [Source_Index, Target_Index, Score]
    n_nodes: Total number of nodes in the network
    params: Dictionary containing execution parameters
    
    Returns:
        pos (np.ndarray): Final coordinates, (n_nodes, LAYOUT_DIMENSIONS)
        box_limit (float): Boundary box size
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP library is not installed. Please install 'umap-learn' to use the UMAP layout engine.")
        
    print(f"Running UMAP global layout on {n_nodes} nodes...")

    layout_seed = _resolve_seed(params)
    dimensions = int(params.get('LAYOUT_DIMENSIONS', 2) or 2)
    if dimensions not in (2, 3):
        raise ValueError(
            f"LAYOUT_DIMENSIONS must be 2 or 3, got {dimensions!r}."
        )
    
    target_box = (np.sqrt(n_nodes) * 2.5 + 5.0)
    final_box_limit = target_box * params.get('BOX_SCALE', 1.0)

    connectivity = np.asarray(connectivity).reshape(-1, 3)
    # An edge needs two different nodes: a network of self-loops only has none.
    if not np.any(connectivity[:, 0] != connectivity[:, 1]):
        # Edge case: No edges at all
        print("Warning: Network has no edges. Generating random layout.")
        pos = np.random.default_rng(layout_seed).uniform(
            -final_box_limit / 2.0, final_box_limit / 2.0, (n_nodes, dimensions)
        ).astype(np.float32)
        return pos, final_box_limit

    # A self-loop is no edge: drop it before the best score is taken, or a loop that scores
    # above every real edge would shift all the distances.
    connectivity = connectivity[connectivity[:, 0] != connectivity[:, 1]]

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

    knn_indices, knn_dists = build_precomputed_knn(
        sources, targets, distances, n_nodes, real_neighbors
    )

    # --- 3. Run UMAP with Native precomputed_knn ---
    print(f"  > Initializing UMAP (K={requested_neighbors} other nodes, "
          f"up to {real_neighbors} available + self, min_dist={min_dist})")
    warnings.filterwarnings('ignore', category=UserWarning, module='umap')
    X_dummy = np.zeros((n_nodes, 1), dtype=np.float32)
    with warnings.catch_warnings(), _seeded_arpack(layout_seed):
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=dimensions,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            precomputed_knn=(knn_indices, knn_dists, None),
            init='random' if n_nodes <= 3 else 'spectral',
            random_state=layout_seed # LAYOUT_SEED; None opts out of determinism
        )
        final_pos = reducer.fit_transform(X_dummy).astype(np.float32)

    # --- 4. Scale and Center the Connected Vertices ---
    # A vertex with no edge has no coordinates (NaN). The centre and the scale come from the
    # connected vertices alone, so the others cannot move or stretch the layout.
    isolated = ~np.isfinite(final_pos).all(axis=1)
    connected_mask = ~isolated
    if np.any(connected_mask):
        connected = final_pos[connected_mask]
        center = (np.max(connected, axis=0) + np.min(connected, axis=0)) / 2.0
        connected -= center
        umap_max_spread = float(np.max(np.ptp(connected, axis=0))) + 1e-9  # avoid division by zero
        # Two connected vertices, or vertices that coincide, have no spread to scale by.
        scale_factor = (final_box_limit * 1.6) / umap_max_spread if umap_max_spread > 1e-6 else 1.0
        final_pos[connected_mask] = connected * scale_factor
    else:
        scale_factor = final_box_limit * 1.6

    # --- 5. Place the Disconnected Vertices ---
    # A seeded random clump in the unit box, centred on the middle of the layout, at the scale
    # of the connected vertices but never wider than a fifth of the box.
    if np.any(isolated):
        rng = np.random.RandomState(layout_seed)
        clump = rng.uniform(-0.5, 0.5, (np.sum(isolated), dimensions)).astype(np.float32)
        final_pos[isolated] = clump * min(scale_factor, 0.2 * final_box_limit)

    print("UMAP layout calculation complete.")
    return final_pos, final_box_limit
