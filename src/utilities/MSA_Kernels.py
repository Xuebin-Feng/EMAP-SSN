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

"""Numba-compiled CPU kernels for Embedding_MSA guide trees and profile alignment.

Keeping cached ``@jit`` kernels in this canonical ``utilities.MSA_Kernels``
module guarantees a single stable Numba cache module name whether
``Embedding_MSA.py`` is run as ``__main__``, imported as ``Embedding_MSA``,
imported as ``tools.Embedding_MSA``, or loaded dynamically in tests.
"""

from __future__ import annotations

import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True, cache=True)
def populate_condensed_matrix(D_condensed, num_seqs, edge_i, edge_j, edge_dists):
    """Blazing fast population of the 1D condensed array from sparse data."""
    for k in range(len(edge_i)):
        i = edge_i[k]
        j = edge_j[k]
        if i > j:
            temp = i
            i = j
            j = temp
        idx = int(num_seqs * i - i * (i + 1) / 2 + j - i - 1)
        D_condensed[idx] = edge_dists[k]


@jit(nopython=True, cache=True)
def _prep_tree_euler(Z, num_seqs):
    """Build O(N) Euler-tour leaf spans, parent, sibling, max_leaf, and merge heights."""
    total_nodes = 2 * num_seqs - 1
    left = np.full(total_nodes, -1, dtype=np.int32)
    right = np.full(total_nodes, -1, dtype=np.int32)
    parent = np.full(total_nodes, -1, dtype=np.int32)
    sibling = np.full(total_nodes, -1, dtype=np.int32)
    max_leaf = np.empty(total_nodes, dtype=np.int32)
    merge_h = np.zeros(total_nodes, dtype=np.float32)

    for i in range(num_seqs):
        max_leaf[i] = i

    for step in range(num_seqs - 1):
        a = int(Z[step, 0])
        b = int(Z[step, 1])
        p = num_seqs + step
        left[p] = a
        right[p] = b
        parent[a] = p
        parent[b] = p
        sibling[a] = b
        sibling[b] = a
        ma = max_leaf[a]
        mb = max_leaf[b]
        max_leaf[p] = ma if ma > mb else mb
        merge_h[p] = np.float32(Z[step, 2])

    sub_start = np.empty(total_nodes, dtype=np.int32)
    sub_end = np.empty(total_nodes, dtype=np.int32)
    dfs_leaves = np.empty(num_seqs, dtype=np.int32)

    stack_node = np.empty(total_nodes * 2, dtype=np.int32)
    stack_state = np.empty(total_nodes * 2, dtype=np.int8)
    root = total_nodes - 1
    top = 0
    stack_node[top] = root
    stack_state[top] = 0
    top += 1
    cursor = 0

    while top > 0:
        top -= 1
        node = stack_node[top]
        state = stack_state[top]
        if state == 0:
            sub_start[node] = cursor
            if node < num_seqs:
                dfs_leaves[cursor] = node
                cursor += 1
                sub_end[node] = cursor
            else:
                stack_node[top] = node
                stack_state[top] = 1
                top += 1
                stack_node[top] = right[node]
                stack_state[top] = 0
                top += 1
                stack_node[top] = left[node]
                stack_state[top] = 0
                top += 1
        else:
            sub_end[node] = cursor

    return parent, sibling, max_leaf, merge_h, sub_start, sub_end, dfs_leaves


@jit(nopython=True, cache=True)
def _prep_tree_euler_nj(Z, num_seqs, node_length):
    """Build O(N) Euler-tour spans + float64 distance-to-root for NJ path lengths."""
    parent, sibling, max_leaf, _mh, sub_start, sub_end, dfs_leaves = _prep_tree_euler(
        Z, num_seqs
    )
    total_nodes = 2 * num_seqs - 1
    root_dist = np.zeros(total_nodes, dtype=np.float64)
    for step in range(num_seqs - 2, -1, -1):
        p = num_seqs + step
        a = int(Z[step, 0])
        b = int(Z[step, 1])
        dp = root_dist[p]
        root_dist[a] = dp + node_length[a]
        root_dist[b] = dp + node_length[b]
    return parent, sibling, max_leaf, root_dist, sub_start, sub_end, dfs_leaves


@jit(nopython=True, parallel=True, cache=True)
def _accumulate_cophenetic_batch_kernel(
    parents,
    siblings,
    max_leaves,
    merge_hs,
    sub_starts,
    sub_ends,
    dfs_leaves_all,
    num_seqs,
    coph_accumulator,
):
    """Add K UPGMA trees in exact tree order 0..K-1 via L1-resident row buffers."""
    n_trees = parents.shape[0]
    n_rows = num_seqs - 1
    half = (n_rows + 1) // 2
    for t_idx in prange(half):
        first = np.int64(t_idx)
        for side in range(2):
            i = first if side == 0 else np.int64(n_rows - 1) - first
            if side == 1 and i == first:
                break
            row_len = num_seqs - 1 - i
            row_off = num_seqs * i - (i * (i + 1)) // 2
            buf = np.empty(row_len, dtype=np.float32)
            for c in range(row_len):
                buf[c] = coph_accumulator[row_off + c]

            for tr in range(n_trees):
                par = parents[tr]
                sib_arr = siblings[tr]
                ml = max_leaves[tr]
                mh = merge_hs[tr]
                ss = sub_starts[tr]
                se = sub_ends[tr]
                leaves = dfs_leaves_all[tr]

                cur = i
                p = par[cur]
                while p != -1:
                    sib = sib_arr[cur]
                    if ml[sib] > i:
                        h = mh[p]
                        s0 = ss[sib]
                        s1 = se[sib]
                        for pos in range(s0, s1):
                            j = leaves[pos]
                            if j > i:
                                buf[j - i - 1] += h
                    cur = p
                    p = par[cur]

            for c in range(row_len):
                coph_accumulator[row_off + c] = buf[c]


@jit(nopython=True, parallel=True, cache=True)
def _accumulate_path_lengths_batch_kernel(
    parents,
    siblings,
    max_leaves,
    root_dists,
    sub_starts,
    sub_ends,
    dfs_leaves_all,
    num_seqs,
    path_accumulator,
):
    """Add K NJ trees in exact tree order 0..K-1 via L1-resident row buffers."""
    n_trees = parents.shape[0]
    n_rows = num_seqs - 1
    half = (n_rows + 1) // 2
    for t_idx in prange(half):
        first = np.int64(t_idx)
        for side in range(2):
            i = first if side == 0 else np.int64(n_rows - 1) - first
            if side == 1 and i == first:
                break
            row_len = num_seqs - 1 - i
            row_off = num_seqs * i - (i * (i + 1)) // 2
            buf = np.empty(row_len, dtype=np.float32)
            for c in range(row_len):
                buf[c] = path_accumulator[row_off + c]

            for tr in range(n_trees):
                par = parents[tr]
                sib_arr = siblings[tr]
                ml = max_leaves[tr]
                rd = root_dists[tr]
                ss = sub_starts[tr]
                se = sub_ends[tr]
                leaves = dfs_leaves_all[tr]

                di = rd[i]
                cur = i
                p = par[cur]
                while p != -1:
                    sib = sib_arr[cur]
                    if ml[sib] > i:
                        base = di - 2.0 * rd[p]
                        s0 = ss[sib]
                        s1 = se[sib]
                        for pos in range(s0, s1):
                            j = leaves[pos]
                            if j > i:
                                buf[j - i - 1] += np.float32(base + rd[j])
                    cur = p
                    p = par[cur]

            for c in range(row_len):
                path_accumulator[row_off + c] = buf[c]


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def _nj_dense_row_minima(D, slot_node_id, r_slot, row_dmin, k, row_min, row_arg):
    """Unit-stride SIMD row-minimum Q search on shrinking k x k active matrix."""
    r_max = -1e300
    for s in range(k):
        if r_slot[s] > r_max:
            r_max = r_slot[s]

    global_ub = 1e15
    if k >= 64:
        top_s = np.empty(4, dtype=np.int64)
        top_v = np.full(4, -1e300, dtype=np.float64)
        for s in range(k):
            rv = r_slot[s]
            if rv > top_v[3]:
                pos = 3
                while pos > 0 and rv > top_v[pos - 1]:
                    top_v[pos] = top_v[pos - 1]
                    top_s[pos] = top_s[pos - 1]
                    pos -= 1
                top_v[pos] = rv
                top_s[pos] = s
        for ti in range(4):
            su = top_s[ti]
            ru = r_slot[su]
            for sv in range(k):
                if sv != su:
                    q = D[su, sv] - (ru + r_slot[sv])
                    if q < global_ub:
                        global_ub = q

    for su in prange(k):
        ru = r_slot[su]
        if k >= 64 and row_dmin[su] - (ru + r_max) > global_ub + 1e-9:
            row_min[su] = 1e15
            row_arg[su] = -1
            continue
        u_id = slot_node_id[su]
        best_q = 1e15
        best_sv = -1
        best_vid = 2147483647
        for sv in range(k):
            v_id = slot_node_id[sv]
            if v_id > u_id:
                q = D[su, sv] - (ru + r_slot[sv])
                if q < best_q or (q == best_q and v_id < best_vid):
                    best_q = q
                    best_sv = sv
                    best_vid = v_id
        row_min[su] = best_q
        row_arg[su] = best_sv


@jit(nopython=True, fastmath=True, cache=True)
def neighbor_joining_compact_kernel(D_compact):
    """Exact Saitou-Nei NJ in a compact shrinking (k x k) submatrix of (N, N) float64.

    Uses 8 N^2 B RAM (4x less than (2N-1)x(2N-1)), keeps all active rows/cols
    strictly contiguous in D[0:k, 0:k] for unit-stride SIMD and cache residency,
    uses O(1) row lower-bound pruning, and preserves exact ascending (u, v) tie order.
    """
    N = D_compact.shape[0]
    Z = np.zeros((N - 1, 4), dtype=np.float64)
    branch_lengths = np.zeros((N - 1, 2), dtype=np.float64)

    slot_node_id = np.arange(N, dtype=np.int32)
    k = N

    R = np.zeros(N, dtype=np.float64)
    row_dmin = np.empty(N, dtype=np.float64)
    for i in range(N):
        s = 0.0
        dm = 1e300
        for j in range(N):
            v = D_compact[i, j]
            s += v
            if j != i and v < dm:
                dm = v
        R[i] = s
        row_dmin[i] = dm

    node_height = np.zeros(2 * N - 1, dtype=np.float64)
    num_leaves = np.ones(2 * N - 1, dtype=np.float64)
    r_slot = np.empty(N, dtype=np.float64)
    row_min = np.empty(N, dtype=np.float64)
    row_arg = np.empty(N, dtype=np.int64)

    for step in range(N - 1):
        if k > 2:
            inv_k_minus_2 = 1.0 / (k - 2)
            for s in range(k):
                r_slot[s] = R[s] * inv_k_minus_2

            _nj_dense_row_minima(
                D_compact, slot_node_id, r_slot, row_dmin, k, row_min, row_arg
            )
            min_Q = 1e15
            su = -1
            sv = -1
            best_uid = 2147483647
            for s in range(k):
                q = row_min[s]
                uid = slot_node_id[s]
                if q < min_Q or (q == min_Q and uid < best_uid):
                    min_Q = q
                    su = s
                    sv = row_arg[s]
                    best_uid = uid
        else:
            if slot_node_id[0] < slot_node_id[1]:
                su, sv = 0, 1
            else:
                su, sv = 1, 0

        u = slot_node_id[su]
        v = slot_node_id[sv]
        w = N + step

        dist_uv = D_compact[su, sv]
        child_max = max(node_height[u], node_height[v])
        node_height[w] = max(dist_uv, child_max)
        if k > 2:
            length_u = 0.5 * (dist_uv + r_slot[su] - r_slot[sv])
        else:
            length_u = 0.5 * dist_uv
        length_v = dist_uv - length_u

        sum_d_wm = 0.0
        w_dmin = 1e300
        for m in range(k):
            if m != su and m != sv:
                d_um = D_compact[su, m]
                d_vm = D_compact[sv, m]
                d_wm = 0.5 * (d_um + d_vm - dist_uv)
                D_compact[su, m] = d_wm
                D_compact[m, su] = d_wm
                sum_d_wm += d_wm
                if d_wm < w_dmin:
                    w_dmin = d_wm
                R[m] = R[m] - d_um - d_vm + d_wm
                if d_wm < row_dmin[m]:
                    row_dmin[m] = d_wm
        D_compact[su, su] = 0.0
        R[su] = sum_d_wm
        row_dmin[su] = w_dmin
        slot_node_id[su] = w

        c1 = u if u < v else v
        c2 = v if u < v else u
        Z[step, 0] = float(c1)
        Z[step, 1] = float(c2)
        Z[step, 2] = node_height[w]
        Z[step, 3] = num_leaves[u] + num_leaves[v]
        if u < v:
            branch_lengths[step, 0] = length_u
            branch_lengths[step, 1] = length_v
        else:
            branch_lengths[step, 0] = length_v
            branch_lengths[step, 1] = length_u
        num_leaves[w] = num_leaves[u] + num_leaves[v]

        last = k - 1
        if sv != last:
            for m in range(last):
                val = D_compact[last, m]
                D_compact[sv, m] = val
                D_compact[m, sv] = val
            D_compact[sv, sv] = 0.0
            R[sv] = R[last]
            row_dmin[sv] = row_dmin[last]
            slot_node_id[sv] = slot_node_id[last]
        k -= 1

    return Z, branch_lengths


@jit(nopython=True, cache=True)
def linkage_split_keys(Z, num_seqs, leaf_keys):
    """Return a 64-bit key for each non-trivial split of a linkage tree.

    A clade's key XORs its leaves' keys, so a clade and its complement differ
    by the key of all leaves. The smaller of the two names the split wherever
    the tree's root sits. Clades of all leaves but one split off one leaf and
    are skipped. The root's two children name the same split twice.
    """
    total = np.uint64(0)
    for leaf in range(num_seqs):
        total ^= leaf_keys[leaf]
    node_keys = np.zeros(2 * num_seqs - 1, dtype=np.uint64)
    node_keys[:num_seqs] = leaf_keys
    keys = np.empty(max(num_seqs - 1, 0), dtype=np.uint64)
    count = 0
    for step in range(num_seqs - 1):
        key = node_keys[int(Z[step, 0])] ^ node_keys[int(Z[step, 1])]
        node_keys[num_seqs + step] = key
        if Z[step, 3] < num_seqs - 1:
            keys[count] = min(key, key ^ total)
            count += 1
    return keys[:count]


@jit(nopython=True, fastmath=True, cache=True)
def calculate_normalized_scores_kernel(
    edge_i, edge_j, raw_scores, align_lens, seq_lens, is_evalue, mode_int
):
    """C-speed kernel for processing hundreds of millions of edge normalizations."""
    num_edges = len(edge_i)
    norm_scores = np.zeros(num_edges, dtype=np.float32)
    max_norm_score = 0.0

    for k in range(num_edges):
        if is_evalue:
            norm_score = raw_scores[k]
        else:
            if mode_int == 0:  # alignment_length
                denom = max(align_lens[k], 1.0)
            else:
                len_src = seq_lens[edge_i[k]]
                len_dst = seq_lens[edge_j[k]]

                if mode_int == 1:  # shorter_sequence
                    denom = min(len_src, len_dst)
                elif mode_int == 2:  # longer_sequence
                    denom = max(len_src, len_dst)
                elif mode_int == 3:  # average_sequence
                    denom = (len_src + len_dst) / 2.0
                else:
                    denom = 1.0

            denom = max(denom, 1e-6)
            norm_score = raw_scores[k] / denom

        norm_scores[k] = norm_score
        if norm_score > max_norm_score:
            max_norm_score = norm_score

    return norm_scores, max_norm_score


@jit(nopython=True, fastmath=True, cache=True)
def run_global_traceback(score_matrix, gap_open, gap_extend):
    """3-state Gotoh affine global alignment using 2-row L1 DP + 1 B/cell packed pointers."""
    N, M = score_matrix.shape
    neg_inf = np.float32(-1e9)
    go = np.float32(gap_open)
    ge = np.float32(gap_extend)

    prev_M = np.full(M + 1, neg_inf, dtype=np.float32)
    prev_D = np.full(M + 1, neg_inf, dtype=np.float32)
    prev_I = np.full(M + 1, neg_inf, dtype=np.float32)
    curr_M = np.empty(M + 1, dtype=np.float32)
    curr_D = np.empty(M + 1, dtype=np.float32)
    curr_I = np.empty(M + 1, dtype=np.float32)

    # Bit-packed traceback pointers (1 byte/cell vs 15 bytes/cell for 6 2D arrays):
    #   bits 0..1: ptr_M (1=M, 2=D, 3=I)
    #   bit 2:     ptr_D (0=M, 1=D)
    #   bit 3:     ptr_I (0=M, 1=I)
    ptr = np.zeros((N + 1, M + 1), dtype=np.uint8)

    prev_M[0] = np.float32(0.0)
    if M >= 1:
        prev_I[1] = go
        for j in range(2, M + 1):
            prev_I[j] = prev_I[j - 1] + ge
            ptr[0, j] = np.uint8(8)

    for i in range(1, N + 1):
        curr_M[0] = neg_inf
        curr_I[0] = neg_inf
        if i == 1:
            curr_D[0] = go
        else:
            curr_D[0] = prev_D[0] + ge
            ptr[i, 0] = np.uint8(4)

        left_M = neg_inf
        left_I = neg_inf
        diag_M = prev_M[0]
        diag_D = prev_D[0]
        diag_I = prev_I[0]

        for j in range(1, M + 1):
            up_M = prev_M[j]
            up_D = prev_D[j]
            up_I = prev_I[j]

            d_from_m = up_M + go
            d_from_d = up_D + ge
            if d_from_m >= d_from_d:
                best_D = d_from_m
                pd_bit = np.uint8(0)
            else:
                best_D = d_from_d
                pd_bit = np.uint8(4)
            curr_D[j] = best_D

            i_from_m = left_M + go
            i_from_i = left_I + ge
            if i_from_m >= i_from_i:
                best_I = i_from_m
                pi_bit = np.uint8(0)
            else:
                best_I = i_from_i
                pi_bit = np.uint8(8)
            curr_I[j] = best_I

            score = score_matrix[i - 1, j - 1]
            m_from_m = diag_M + score
            m_from_d = diag_D + score
            m_from_i = diag_I + score
            best_M = m_from_m
            pm_bits = np.uint8(1)
            if m_from_d > best_M:
                best_M = m_from_d
                pm_bits = np.uint8(2)
            if m_from_i > best_M:
                best_M = m_from_i
                pm_bits = np.uint8(3)
            curr_M[j] = best_M

            ptr[i, j] = pm_bits | pd_bit | pi_bit

            diag_M = up_M
            diag_D = up_D
            diag_I = up_I
            left_M = best_M
            left_I = best_I

        prev_M, curr_M = curr_M, prev_M
        prev_D, curr_D = curr_D, prev_D
        prev_I, curr_I = curr_I, prev_I

    path_buffer = np.zeros(N + M, dtype=np.int8)
    k = 0
    i = N
    j = M
    best_final = prev_M[M]
    state = 1
    if prev_D[M] > best_final:
        best_final = prev_D[M]
        state = 2
    if prev_I[M] > best_final:
        best_final = prev_I[M]
        state = 3

    while i > 0 or j > 0:
        packed = ptr[i, j]
        if state == 1:
            path_buffer[k] = 1
            k += 1
            state = int(packed & np.uint8(3))
            i -= 1
            j -= 1
        elif state == 2:
            path_buffer[k] = 2
            k += 1
            state = 2 if (packed & np.uint8(4)) else 1
            i -= 1
        elif state == 3:
            path_buffer[k] = 3
            k += 1
            state = 3 if (packed & np.uint8(8)) else 1
            j -= 1
        else:
            break

    return path_buffer[:k]
