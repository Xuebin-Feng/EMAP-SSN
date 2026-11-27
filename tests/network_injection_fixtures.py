# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Reference cut-rule oracles and world builders shared by Network_Injection tests."""

import os
import sys

import h5py
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utilities import Prefilter_Calibration  # noqa: E402


def world_similarity(embeddings, headers):
    """The shared similarity of the new set: max-pooled, length-adjusted cosine (float64, n x n)."""
    lengths = [len(embeddings[h]) for h in headers]
    return Prefilter_Calibration.pair_similarity(embeddings, headers, lengths)


def oracle_cut_rule(old_headers, old_i, old_j, new_headers, s, calibration):
    """What the cut rule decides, by plain loops. Returns a dict.

    Keys: sparse, cached, required (sets of pair keys i * N + j), K, s_min,
    lowest (first edge with the strictly lowest s, -1 if none), t_prime, cut,
    set_by, kept_after (share of pairs with s >= cut). calibration is the set
    of edge numbers that are recorded calibration edges.
    """
    new_idx = {h: idx for idx, h in enumerate(new_headers)}
    new_n = len(new_headers)
    old_n = len(old_headers)
    total_old = old_n * (old_n - 1) // 2
    sparse = len(old_i) < total_old
    old_set = set(old_headers)
    cached = set()
    for k in range(len(old_i)):
        u_h, v_h = old_headers[old_i[k]], old_headers[old_j[k]]
        if u_h in new_idx and v_h in new_idx:
            u, v = new_idx[u_h], new_idx[v_h]
            cached.add(min(u, v) * new_n + max(u, v))
    pairs = [(i, j) for i in range(new_n) for j in range(i + 1, new_n)]
    info = {"sparse": sparse, "cached": cached, "K": None, "s_min": None, "lowest": -1,
            "t_prime": None, "cut": None, "set_by": None, "kept_after": None}
    if not sparse or len(old_i) == 0:
        # A complete input, or a sparse one with no edge at all (nothing to compare with, as
        # on main): every pair with a new sequence, plus the old edges; no other old-old pair.
        info["required"] = set(cached) | {
            i * new_n + j for i, j in pairs
            if not (new_headers[i] in old_set and new_headers[j] in old_set)
        }
        return info
    info["K"] = (len(old_i) - len(calibration)) / total_old
    for k in range(len(old_i)):
        if k in calibration:
            continue
        u_h, v_h = old_headers[old_i[k]], old_headers[old_j[k]]
        if u_h in new_idx and v_h in new_idx:
            value = float(s[new_idx[u_h], new_idx[v_h]])
            if value == value and (info["lowest"] == -1 or value < info["s_min"]):
                info["s_min"], info["lowest"] = value, k
    values = np.array([s[i, j] for i, j in pairs], dtype=np.float64)
    if values.size == 0:
        info["cut"] = float("inf")
    else:
        info["t_prime"] = float(np.percentile(values, 100.0 * (1.0 - info["K"])))
        if info["lowest"] != -1 and info["s_min"] < info["t_prime"]:
            info["cut"], info["set_by"] = info["s_min"], "s_min"
        else:
            info["cut"], info["set_by"] = info["t_prime"], "t'"
    info["required"] = set(cached) | {i * new_n + j for i, j in pairs if s[i, j] >= info["cut"]}
    info["kept_after"] = (
        sum(1 for i, j in pairs if s[i, j] >= info["cut"]) / len(pairs) if pairs else 0.0
    )
    return info


def oracle_pending(new_n, required, cached, computed):
    return [
        i * new_n + j
        for i in range(new_n)
        for j in range(i + 1, new_n)
        if (i * new_n + j) in required
        and (i * new_n + j) not in cached
        and (i * new_n + j) not in computed
    ]


def oracle_merge(new_n, required, cached, new_headers, old_index, old_arrays,
                 actual_idx_map, batches):
    """The original compile_final_output arrays: (i, j, l_score, l_len, g_score, g_len)."""
    old_l_score, old_l_len, old_g_score, old_g_len = old_arrays
    sorted_pairs = sorted(required)
    final = {name: np.zeros(len(sorted_pairs), dtype=dtype) for name, dtype in (
        ("i", np.uint16), ("j", np.uint16), ("l_score", np.float32),
        ("l_len", np.uint16), ("g_score", np.float32), ("g_len", np.uint16))}
    slot = {}
    for idx, pair_id in enumerate(sorted_pairs):
        final["i"][idx] = pair_id // new_n
        final["j"][idx] = pair_id % new_n
        slot[pair_id] = idx
    for pair_id in cached:
        if pair_id in slot:
            idx = slot[pair_id]
            u = old_index[new_headers[pair_id // new_n]]
            v = old_index[new_headers[pair_id % new_n]]
            lo, hi = min(u, v), max(u, v)
            theoretical = lo * len(old_index) - lo * (lo + 1) // 2 + (hi - lo - 1)
            actual = actual_idx_map[theoretical]
            final["l_score"][idx] = old_l_score[actual]
            final["l_len"][idx] = old_l_len[actual]
            final["g_score"][idx] = old_g_score[actual]
            final["g_len"][idx] = old_g_len[actual]
    for arr_i, arr_j, l_score, l_len, g_score, g_len in batches:
        for k in range(len(arr_i)):
            pair_id = min(int(arr_i[k]), int(arr_j[k])) * new_n + max(int(arr_i[k]), int(arr_j[k]))
            if pair_id in slot and pair_id not in cached:
                idx = slot[pair_id]
                final["l_score"][idx] = l_score[k]
                final["l_len"][idx] = l_len[k]
                final["g_score"][idx] = g_score[k]
                final["g_len"][idx] = g_len[k]
    return final


def random_world(rng, kind, large=False):
    """An old network and a new sequence set that drops some old sequences and adds new ones.

    Each sequence has 1 to 4 residue rows drawn from a tiny grid, so that many
    similarities are exactly equal (ties) and the lengths differ.
    kind: "dense" (a complete input), "sparse_record" (a recorded calibration
    sample), "sparse_plain", "sparse_all_dropped" (every old edge loses an end
    in the new set) or "sparse_no_edge". large: about 300 to 380 new sequences,
    so that pair keys (min * N + max) pass 65,535 and wrap in a 16-bit index array.
    Returns (embeddings, old_headers, new_headers, edges, calibration), edges
    being (i, j) with i < j in old indices.
    """
    pool = [f"s{k}" for k in range(int(rng.integers(380, 420) if large else rng.integers(12, 24)))]
    grid = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 1]], np.float32)
    embeddings = {
        h: grid[rng.integers(len(grid), size=int(rng.integers(1, 5)))] for h in pool
    }
    old_n = int(rng.integers(300, 340) if large else rng.integers(3, 10))
    old_headers = [pool[k] for k in rng.permutation(len(pool))[:old_n]]
    all_pairs = [(i, j) for i in range(old_n) for j in range(i + 1, old_n)]
    if kind == "dense":
        edges = [all_pairs[k] for k in rng.permutation(len(all_pairs))]
    elif kind == "sparse_no_edge":
        edges = []
    elif kind == "sparse_all_dropped" and large:
        # Edges among the first 30 old sequences only, so that most sequences survive the drop.
        few = [(i, j) for (i, j) in all_pairs if j < 30]
        edges = [few[k] for k in rng.permutation(len(few))[: int(rng.integers(1, len(few)))]]
    else:
        count = int(rng.integers(1, len(all_pairs)))
        edges = [all_pairs[k] for k in rng.permutation(len(all_pairs))[:count]]
    calibration = set()
    if kind == "sparse_record":
        calibration = {int(k) for k in rng.permutation(len(edges))[: int(rng.integers(1, len(edges) + 1))]}
    kept = [h for h in old_headers if rng.random() < (0.92 if large else 0.75)]
    if kind == "sparse_all_dropped":
        # Keep only sequences that no edge touches (a dropped one for every edge).
        touched = {old_headers[k] for e in edges for k in e}
        kept = [h for h in old_headers if h not in touched]
    others = [h for h in pool if h not in old_headers]
    added = others[: int(rng.integers(30, 41) if large else rng.integers(0, 7))]
    members = kept + added
    if len(members) < 2:
        members = (members + others[len(added):] + old_headers)[:2] if kind == "sparse_all_dropped" else old_headers[:2]
    new_headers = [members[k] for k in rng.permutation(len(members))]
    return embeddings, old_headers, new_headers, edges, calibration


def write_batch_file(results_dir, name, pairs, scores, index_dtype=np.uint32):
    arr_i = np.array([p[0] for p in pairs], dtype=index_dtype)
    arr_j = np.array([p[1] for p in pairs], dtype=index_dtype)
    with h5py.File(os.path.join(results_dir, name), "w") as hf:
        hf.create_dataset("i", data=arr_i)
        hf.create_dataset("j", data=arr_j)
        hf.create_dataset("l_score", data=np.array([s[0] for s in scores], np.float32))
        hf.create_dataset("l_len", data=np.array([s[1] for s in scores], np.uint16))
        hf.create_dataset("g_score", data=np.array([s[2] for s in scores], np.float32))
        hf.create_dataset("g_len", data=np.array([s[3] for s in scores], np.uint16))
