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

"""PaCMAP layout of a network, on Numba threads.

A port of the optimiser of PaCMAP (Wang, Huang, Rudin and Shaposhnik, "Understanding How
Dimension Reduction Tools Work", JMLR 22(201), 2021; reference implementation: the
``pacmap`` package, Apache-2.0) to the pairs that utilities.PaCMAP_Pairs builds from a
network's scores. The loss and the schedule are the package's:

* with d = |y_i - y_j|^2 + 1, a near pair costs w * d / (10 + d), a mid-near pair
  w * d / (10000 + d), and a further pair w * 1 / (1 + d);
* 450 iterations of Adam (learning rate 1, betas 0.9 and 0.999), in three phases: for
  100 iterations the mid-near weight falls linearly from 1000 to 3 (near weight 2), for
  100 more it stays 3 (near weight 3), then it is 0 and the near weight 1; the further
  weight is 1 throughout. The start is a random normal cloud scaled by 1e-4.

What differs from the package is the gradient. Every node sums the pairs it is in (from
incidence lists), so each pair is evaluated from both ends and no thread writes to a
node another thread reads or writes. A node adds its pairs in a fixed order, so the
coordinates are bit-identical whatever the number of threads. The package's threads
add into shared memory in whatever order they finish, so its own results differ from
run to run, and these differ from the package's in the last digits: equal in
distribution, not in value (tests/test_layout_engine_pacmap.py).

The contract is the UMAP engine's: ``calculate_layout(connectivity, n_nodes, params)``
returns the coordinates, centred and scaled into the box, and the box size.
"""

import math
import time

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        def decorator(function):
            return function
        return decorator

try:
    from utilities import Numba_Threads
    from utilities import PaCMAP_Pairs as pairs
except ImportError:
    import Numba_Threads
    import PaCMAP_Pairs as pairs


# The schedule and the optimiser, as in the package.
ITERATIONS = (100, 100, 250)
INITIAL_MID_NEAR_WEIGHT = 1000.0
LEARNING_RATE = 1.0
BETA_1 = 0.9
BETA_2 = 0.999
ADAM_EPSILON = 1e-7
START_SCALE = 1e-4
DEFAULT_NEIGHBORS = 10


def _resolve_seed(params):
    """Return the layout seed, or None for non-reproducible behaviour.

    Kept behaviourally identical to Layout_Engine_UMAP._resolve_seed. The engines share no
    imports by design - the only contract between them is
    calculate_layout(connectivity, n_nodes, params) - so the check is duplicated
    rather than imported.
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


def phase_weights(iteration):
    """(mid-near, near, further) weights of an iteration, counted from 0."""
    first, second, _ = ITERATIONS
    if iteration < first:
        progress = iteration / first
        return (1.0 - progress) * INITIAL_MID_NEAR_WEIGHT + progress * 3.0, 2.0, 1.0
    if iteration < first + second:
        return 3.0, 3.0, 1.0
    return 0.0, 1.0, 1.0


@njit(parallel=True, nogil=True, cache=True)
def _gradient(position, near_pointer, near_to, mid_pointer, mid_to, far_pointer, far_to,
              near_weight, mid_weight, far_weight, gradient):
    """The gradient of the loss at every node, from the pairs the node is in.

    Each pair is added by both its nodes, from the same difference with opposite signs, so
    the two sums are exact negatives of each other's term. Nodes only read the coordinates
    and write their own row.
    """
    count, dimensions = position.shape
    for node in prange(count):
        g0 = 0.0
        g1 = 0.0
        g2 = 0.0
        y0 = np.float64(position[node, 0])
        y1 = np.float64(position[node, 1])
        y2 = np.float64(position[node, 2]) if dimensions == 3 else 0.0
        for q in range(near_pointer[node], near_pointer[node + 1]):
            other = near_to[q]
            d0 = y0 - np.float64(position[other, 0])
            d1 = y1 - np.float64(position[other, 1])
            d2 = (y2 - np.float64(position[other, 2])) if dimensions == 3 else 0.0
            distance = 1.0 + d0 * d0 + d1 * d1 + d2 * d2
            weight = near_weight * 20.0 / ((10.0 + distance) * (10.0 + distance))
            g0 += weight * d0
            g1 += weight * d1
            g2 += weight * d2
        if mid_weight != 0.0:
            for q in range(mid_pointer[node], mid_pointer[node + 1]):
                other = mid_to[q]
                d0 = y0 - np.float64(position[other, 0])
                d1 = y1 - np.float64(position[other, 1])
                d2 = (y2 - np.float64(position[other, 2])) if dimensions == 3 else 0.0
                distance = 1.0 + d0 * d0 + d1 * d1 + d2 * d2
                weight = mid_weight * 20000.0 / ((10000.0 + distance) * (10000.0 + distance))
                g0 += weight * d0
                g1 += weight * d1
                g2 += weight * d2
        for q in range(far_pointer[node], far_pointer[node + 1]):
            other = far_to[q]
            d0 = y0 - np.float64(position[other, 0])
            d1 = y1 - np.float64(position[other, 1])
            d2 = (y2 - np.float64(position[other, 2])) if dimensions == 3 else 0.0
            distance = 1.0 + d0 * d0 + d1 * d1 + d2 * d2
            weight = far_weight * 2.0 / ((1.0 + distance) * (1.0 + distance))
            g0 -= weight * d0
            g1 -= weight * d1
            g2 -= weight * d2
        gradient[node, 0] = g0
        gradient[node, 1] = g1
        if dimensions == 3:
            gradient[node, 2] = g2


@njit(parallel=True, nogil=True, cache=True)
def _adam_step(position, gradient, first_moment, second_moment, step_size):
    count, dimensions = position.shape
    for node in prange(count):
        for axis in range(dimensions):
            g = np.float64(gradient[node, axis])
            first = np.float64(first_moment[node, axis])
            second = np.float64(second_moment[node, axis])
            first += (1.0 - BETA_1) * (g - first)
            second += (1.0 - BETA_2) * (g * g - second)
            first_moment[node, axis] = first
            second_moment[node, axis] = second
            position[node, axis] = np.float32(
                np.float64(position[node, axis]) - step_size * first / (math.sqrt(second) + ADAM_EPSILON)
            )


def optimise(count, dimensions, near, mid, far, seed, *, report=None):
    """The PaCMAP optimisation of ``count`` nodes: (count, dimensions) float32 coordinates.

    near, mid and far are the pair sets' incidence lists, each a (pointer, partners) pair.
    """
    position = pairs.normal_start(count, dimensions, seed, START_SCALE)
    gradient = np.zeros_like(position)
    first_moment = np.zeros_like(position)
    second_moment = np.zeros_like(position)
    total = sum(ITERATIONS)
    for iteration in range(total):
        mid_weight, near_weight, far_weight = phase_weights(iteration)
        _gradient(position, near[0], near[1], mid[0], mid[1], far[0], far[1],
                  near_weight, mid_weight, far_weight, gradient)
        step_size = LEARNING_RATE * math.sqrt(1.0 - BETA_2 ** (iteration + 1)) / (1.0 - BETA_1 ** (iteration + 1))
        _adam_step(position, gradient, first_moment, second_moment, step_size)
        if report is not None and (iteration + 1) % 50 == 0:
            report(iteration + 1, total)
    return position


def build_pairs(connectivity, n_nodes, neighbors, seed, mid_near=None):
    """The pair sets of a network among its connected nodes.

    Returns (ids, near, mid, far): the sorted numbers of the connected nodes, and the
    three pair sets as incidence lists over the positions in ``ids``. ``mid_near`` is the
    (pairs, 2) table that utilities.PaCMAP_Pairs.mid_near_pairs found from the stored
    scores, in node numbers; without it the mid-near partners are random nodes.
    """
    sources = connectivity[:, 0].astype(np.int64)
    targets = connectivity[:, 1].astype(np.int64)
    scores = connectivity[:, 2].astype(np.float64)
    ids, compact = pairs.connected_nodes(sources, targets, n_nodes)
    count = len(ids)
    with Numba_Threads.limited_threads(Numba_Threads.default_thread_count()):
        near_table = pairs.near_partners(compact[sources], compact[targets], scores, count, neighbors)
        if mid_near is None:
            mid_near = pairs.random_mid_near_pairs(ids, neighbors, seed)
        mid_near = np.asarray(mid_near, dtype=np.int64).reshape(-1, 2)
        first, second = compact[mid_near[:, 0]], compact[mid_near[:, 1]]
        usable = (first >= 0) & (second >= 0) & (first != second)
        far_table = pairs.further_partners(near_table, count, neighbors, seed)
        near = pairs.incidence_lists(*pairs.rows_of_partners(near_table), count)
        mid = pairs.incidence_lists(first[usable], second[usable], count)
        far = pairs.incidence_lists(*pairs.rows_of_partners(far_table), count)
    return ids, near, mid, far


def calculate_layout(connectivity, n_nodes, params):
    """
    Alternative layout generation pipeline using PaCMAP on a network's strongest edges.

    connectivity: N x 3 NumPy array representing [Source_Index, Target_Index, Score]
    n_nodes: Total number of nodes in the network
    params: Dictionary containing execution parameters. PACMAP_NEIGHBORS is the number of
        near pairs per node (and the number of strongest edges the caller kept per
        node); PACMAP_MID_NEAR optionally holds the mid-near pairs found from the stored
        scores (utilities.PaCMAP_Pairs.mid_near_pairs).

    Returns:
        pos (np.ndarray): Final coordinates, (n_nodes, LAYOUT_DIMENSIONS)
        box_limit (float): Boundary box size
    """
    if not NUMBA_AVAILABLE:
        raise ImportError("Numba is not installed. Please install 'numba' to use the PaCMAP layout engine.")

    print(f"Running PaCMAP global layout on {n_nodes} nodes...")

    layout_seed = _resolve_seed(params)
    dimensions = int(params.get('LAYOUT_DIMENSIONS', 2) or 2)
    if dimensions not in (2, 3):
        raise ValueError(
            f"LAYOUT_DIMENSIONS must be 2 or 3, got {dimensions!r}."
        )
    neighbors = int(params.get('PACMAP_NEIGHBORS', DEFAULT_NEIGHBORS) or DEFAULT_NEIGHBORS)
    if neighbors < 1:
        raise ValueError(f"PACMAP_NEIGHBORS must be at least 1, got {neighbors!r}.")

    target_box = (np.sqrt(n_nodes) * 2.5 + 5.0)
    final_box_limit = target_box * params.get('BOX_SCALE', 1.0)

    connectivity = np.asarray(connectivity).reshape(-1, 3)
    has_edge = connectivity.shape[0] > 0 and bool(np.any(connectivity[:, 0] != connectivity[:, 1]))
    if not has_edge:
        # Edge case: No edges at all
        print("Warning: Network has no edges. Generating random layout.")
        pos = np.random.default_rng(layout_seed).uniform(
            -final_box_limit / 2.0, final_box_limit / 2.0, (n_nodes, dimensions)
        ).astype(np.float32)
        return pos, final_box_limit

    seed = pairs.random_seed() if layout_seed is None else layout_seed
    mid_near = params.get('PACMAP_MID_NEAR')
    if mid_near is None:
        print("Warning: no stored pair scores were supplied, so the mid-near partners are random "
              "and the layout keeps its local structure only.")

    started = time.perf_counter()
    ids, near, mid, far = build_pairs(connectivity, n_nodes, neighbors, seed, mid_near)
    count = len(ids)
    print(f"  > Initializing PaCMAP (K={neighbors} near pairs per node; {len(near[1]) // 2} near, "
          f"{len(mid[1]) // 2} mid-near and {len(far[1]) // 2} further pairs among {count} connected nodes)")
    with Numba_Threads.limited_threads(Numba_Threads.default_thread_count()):
        embedding = optimise(count, dimensions, near, mid, far, seed)
    print(f"  > Optimised in {time.perf_counter() - started:.1f} s")

    final_pos = np.empty((n_nodes, dimensions), dtype=np.float32)
    nan_mask = np.ones(n_nodes, dtype=bool)
    nan_mask[ids] = False
    final_pos[ids] = embedding

    # --- Handle Disconnected Vertices (no edge to any node) ---
    # The rule of the UMAP engine: a seeded random place in the unit box around the middle
    # of the layout, and no part in the scaling below.
    if np.any(nan_mask):
        rng = np.random.RandomState(layout_seed)
        final_pos[nan_mask] = rng.uniform(-0.5, 0.5, (np.sum(nan_mask), dimensions)).astype(np.float32)

    # --- Scale and Center Output ---
    connected_mask = ~nan_mask
    global_min = np.min(final_pos[connected_mask], axis=0)
    global_max = np.max(final_pos[connected_mask], axis=0)
    center = (global_max + global_min) / 2.0
    final_pos -= center
    ptp = np.ptp(final_pos[connected_mask], axis=0)
    spread = float(np.max(ptp)) + 1e-9  # avoid division by zero

    scale_factor = (final_box_limit * 1.6) / spread
    final_pos *= scale_factor

    print("PaCMAP layout calculation complete.")
    return final_pos, final_box_limit
