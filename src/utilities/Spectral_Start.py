# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Spectral starting coordinates for SSN layout components.

A component starts on the eigenvectors of its normalized graph Laplacian for
the smallest eigenvalues after the trivial one: lambda2 and lambda3, plus
lambda4 in 3D. The trivial eigenvector, sqrt(degree), is excluded explicitly.
ARPACK's ``which="SM"`` stopping test practically never accepts an eigenvalue
of exactly zero, so it often returned lambda2 to lambda4, and the earlier rule
of skipping the first column laid such components out on lambda3/lambda4.

Components of ``LOBPCG_MIN_NODES`` or more nodes are solved with SciPy's
LOBPCG, constrained away from sqrt(degree), with the Laplacian products on
Numba threads, and preconditioned with an algebraic multigrid V-cycle
(``amg_preconditioner``). With SSN_SPECTRAL_AMG=0, or when that solve fails or
is rejected, LOBPCG runs unpreconditioned from the same start block, exactly as
it did before the preconditioner was added. A result that misses the residual
or orthogonality check falls back to ARPACK, and smaller components keep ARPACK.
``asinh_scaled`` then spreads each axis, so a few outlying nodes no longer
squeeze the rest of a component into a dot.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional
import warnings

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from utilities import Numba_Threads
except ImportError:
    import Numba_Threads


# Module aliasing to prevent duplicate JIT caches in long-running processes
_THIS_MODULE = sys.modules[__name__]
if __name__ == "Spectral_Start":
    sys.modules.setdefault("utilities.Spectral_Start", _THIS_MODULE)
else:
    sys.modules.setdefault("Spectral_Start", _THIS_MODULE)


LOBPCG_MIN_NODES = 2000
# Extra block vectors beyond the wanted axes. With none, lambda3 converged
# slowly or not at all on the 44K network, whose lambda4 lies close above it.
LOBPCG_EXTRA_VECTORS = 4
LOBPCG_TOLERANCE = 1e-6
LOBPCG_MAX_ITERATIONS = 1000
ACCEPTED_RESIDUAL = 1e-5
ACCEPTED_TRIVIAL_COSINE = 1e-6
# An ARPACK vector this parallel to sqrt(degree) is the trivial eigenvector.
# Unit vectors that are mutually orthogonal can have at most one such vector.
TRIVIAL_COSINE = 0.9
# A central 90% range below this share of the full range is numerically empty.
EMPTY_CENTRE_SHARE = 1e-6

# Setting SSN_SPECTRAL_AMG=0 skips the AMG preconditioner, so the layouts made
# by plain LOBPCG can still be reproduced.
AMG_SWITCH_VARIABLE = "SSN_SPECTRAL_AMG"
_OFF_VALUES = {"0", "off", "false", "no"}
# Coarsening stops at this many rows, and the coarsest level is solved
# densely, which stays cheap at this size.
AMG_MAX_COARSE = 500
# A coarsest level this much larger than AMG_MAX_COARSE means coarsening
# stalled; solving it densely would cost too much, so AMG is not used.
AMG_COARSEST_LIMIT = 4 * AMG_MAX_COARSE
# Levels at most, the finest included.
AMG_MAX_LEVELS = 10


def normalized_laplacian(local_edges, scores, node_count):
    """Return the component's normalized Laplacian (COO) and sqrt(degree).

    The degrees are the ones scipy uses to normalize, so the Laplacian times
    sqrt(degree) is zero.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import laplacian

    row = np.concatenate((local_edges[:, 0], local_edges[:, 1]))
    col = np.concatenate((local_edges[:, 1], local_edges[:, 0]))
    data = np.concatenate((scores, scores))
    adjacency = sp.coo_matrix((data, (row, col)), shape=(node_count, node_count))
    return laplacian(adjacency, normed=True, return_diag=True)


def _trivial_cosines(vectors, root_degree):
    unit = root_degree / np.linalg.norm(root_degree)
    return np.abs(unit @ vectors) / np.linalg.norm(vectors, axis=0)


def intended_axes(values, vectors, root_degree, dimensions):
    """Drop the eigenvector parallel to sqrt(degree), if present, and return the
    next ``dimensions`` eigenvectors in ascending eigenvalue order."""
    if not (np.isfinite(values).all() and np.isfinite(vectors).all()):
        raise ValueError("The spectral solver returned non-finite values.")
    vectors = vectors[:, np.argsort(values, kind="stable")]
    kept = np.flatnonzero(_trivial_cosines(vectors, root_degree) <= TRIVIAL_COSINE)
    if len(kept) < dimensions:
        raise ValueError("The spectral solver returned too few non-trivial eigenvectors.")
    return vectors[:, kept[:dimensions]]


@njit(parallel=True, cache=True)
def _csr_rows_product(indptr, indices, data, block, result):
    width = block.shape[1]
    for row in prange(len(indptr) - 1):
        for column in range(width):
            result[row, column] = 0.0
        for position in range(indptr[row], indptr[row + 1]):
            value = data[position]
            neighbour = indices[position]
            for column in range(width):
                result[row, column] += value * block[neighbour, column]


@njit(parallel=True, cache=True)
def _absolute_row_sums(indptr, data, result):
    for row in prange(len(indptr) - 1):
        total = 0.0
        for position in range(indptr[row], indptr[row + 1]):
            total += abs(data[position])
        result[row] = total


@njit(cache=True)
def _standard_aggregation(indptr, indices):
    """Group a symmetric sparsity pattern's nodes into aggregates, serially.

    The greedy passes of pyamg's standard aggregation: in node order, a node
    whose neighbours are all still free starts an aggregate with all of them;
    then every node left over joins the aggregate of its first neighbour from
    that first pass, which always exists. A node without neighbours, which pyamg
    leaves out, becomes an aggregate of its own, numbered after the others.
    Returns each node's aggregate and the number of aggregates.
    """
    n = len(indptr) - 1
    first_pass = np.zeros(n, dtype=np.int64)    # 1-based aggregate, 0 while free
    roots = 0
    for i in range(n):
        if first_pass[i] != 0:
            continue
        has_neighbours = False
        all_free = True
        for position in range(indptr[i], indptr[i + 1]):
            j = indices[position]
            if j != i:
                has_neighbours = True
                if first_pass[j] != 0:
                    all_free = False
                    break
        if has_neighbours and all_free:
            roots += 1
            first_pass[i] = roots
            for position in range(indptr[i], indptr[i + 1]):
                first_pass[indices[position]] = roots
    aggregate = np.empty(n, dtype=np.int64)
    count = roots
    for i in range(n):
        if first_pass[i] != 0:
            aggregate[i] = first_pass[i] - 1
            continue
        joined = 0
        for position in range(indptr[i], indptr[i + 1]):
            if first_pass[indices[position]] != 0:
                joined = first_pass[indices[position]]
                break
        if joined != 0:
            aggregate[i] = joined - 1
        else:
            aggregate[i] = count
            count += 1
    return aggregate, count


def laplacian_operator(csr):
    """A LinearOperator for products with ``csr`` on Numba threads.

    Each row adds its terms in storage order starting from zero, as SciPy's
    own CSR product does: the same bits on x86-64, and within rounding
    elsewhere (an ARM compiler may fuse SciPy's multiply-add). SciPy's
    product is used when Numba is missing.
    """
    from scipy.sparse.linalg import LinearOperator, aslinearoperator

    if not NUMBA_AVAILABLE:
        return aslinearoperator(csr)
    indptr = csr.indptr.astype(np.int64)
    indices = csr.indices
    data = csr.data

    def product(x):
        block = np.ascontiguousarray(x, dtype=np.float64)
        flat = block.ndim == 1
        if flat:
            block = block.reshape(-1, 1)
        result = np.empty_like(block)
        _csr_rows_product(indptr, indices, data, block, result)
        return result.reshape(-1) if flat else result

    return LinearOperator(csr.shape, matvec=product, matmat=product, dtype=np.float64)


def _inverse_absolute_row_sums(csr):
    """1 / (sum of |entries|) per row: l1-Jacobi smoothing weights."""
    if NUMBA_AVAILABLE:
        sums = np.empty(csr.shape[0], dtype=np.float64)
        _absolute_row_sums(csr.indptr.astype(np.int64), csr.data, sums)
    else:
        sums = np.asarray(abs(csr).sum(axis=1), dtype=np.float64).ravel()
    return 1.0 / sums


def amg_disabled():
    """True when SSN_SPECTRAL_AMG asks for LOBPCG without the preconditioner."""
    return os.environ.get(AMG_SWITCH_VARIABLE, "").strip().lower() in _OFF_VALUES


class _BlockVCycle:
    """One symmetric V-cycle over a multigrid hierarchy, for a block of vectors.

    Each level above the coarsest smooths once before and once after its
    coarse correction with l1-Jacobi (weights 1 / row sum of |a|, convergent
    without a spectral-radius estimate); the coarsest level is solved densely
    (see amg_preconditioner). Every product gathers row by row: Numba's rows,
    SciPy's CSR rows, and the restriction stored as CSR rather than applied as
    the prolongator's transpose, so the result does not depend on the thread
    count. A class rather than a recursive closure, so dropping the
    preconditioner frees its arrays at once, without waiting for the garbage
    collector.
    """

    def __init__(self, products, weights, prolongators, restrictions, coarsest):
        self.products = products
        self.weights = weights
        self.prolongators = prolongators
        self.restrictions = restrictions
        self.coarsest = coarsest

    def __call__(self, x):
        block = np.asarray(x, dtype=np.float64)
        flat = block.ndim == 1
        result = self._cycle(0, block.reshape(-1, 1) if flat else block)
        return result.reshape(-1) if flat else result

    def _cycle(self, depth, rhs):
        if depth == len(self.products):
            return self.coarsest @ rhs
        product, weight = self.products[depth], self.weights[depth]
        solution = weight * rhs
        residual = rhs - product.matmat(solution)
        solution += self.prolongators[depth] @ self._cycle(
            depth + 1, self.restrictions[depth] @ residual)
        solution += weight * (rhs - product.matmat(solution))
        return solution


def _tentative_prolongator(aggregate, count, candidate):
    """The prolongator T and the coarse candidate for ``candidate``.

    T[i, aggregate[i]] = candidate[i] / |candidate over that aggregate|, so T
    has orthonormal columns and T @ coarse = candidate, the coarse candidate
    being those norms.
    """
    import scipy.sparse as sp

    norms = np.sqrt(np.bincount(aggregate, weights=candidate * candidate, minlength=count))
    if not (norms > 0.0).all():
        raise RuntimeError("an aggregate has no weight in sqrt(degree)")
    nodes = len(aggregate)
    prolongator = sp.csr_matrix(
        (candidate / norms[aggregate], aggregate, np.arange(nodes + 1)), shape=(nodes, count))
    return prolongator, norms


def aggregation_levels(csr, candidate):
    """The aggregation hierarchy for ``csr`` with near-null vector ``candidate``.

    Returns ([(matrix, prolongator, restriction)] for every level above the
    coarsest, the coarsest matrix, its candidate). Each coarse matrix is the
    Galerkin product restriction @ matrix @ prolongator, with the restriction
    stored as CSR and the indices sorted, so the next aggregation reads a
    canonical pattern.
    """
    levels = []
    matrix = csr
    while matrix.shape[0] > AMG_MAX_COARSE and len(levels) < AMG_MAX_LEVELS - 1:
        aggregate, count = _standard_aggregation(matrix.indptr, matrix.indices)
        if count >= matrix.shape[0]:
            break
        prolongator, coarse_candidate = _tentative_prolongator(aggregate, count, candidate)
        restriction = prolongator.T.tocsr()
        coarse = (restriction @ matrix @ prolongator).tocsr()
        coarse.sort_indices()
        levels.append((matrix, prolongator, restriction))
        matrix, candidate = coarse, coarse_candidate
    if matrix.shape[0] > AMG_COARSEST_LIMIT:
        raise RuntimeError(f"coarsening stalled at {matrix.shape[0]:,} rows")
    return levels, matrix, candidate


def amg_preconditioner(csr, root_degree):
    """A symmetric multigrid V-cycle for ``csr``, applied to whole blocks at once.

    The hierarchy aggregates the Laplacian's own sparsity pattern, with
    sqrt(degree) as the near-null vector and unsmoothed prolongators; nothing
    in it is random. Smoothing the prolongators cost 4-5 s and up to 2 GiB of
    setup on the 44K network, and LOBPCG then needed 128 iterations in 3D
    instead of 56. The cycle treats the whole block at once: pyamg's own
    preconditioner, one column at a time with serial Gauss-Seidel sweeps, was
    slower there than LOBPCG without a preconditioner.
    """
    from scipy.sparse.linalg import LinearOperator

    levels, coarsest_matrix, candidate = aggregation_levels(csr, root_degree.astype(np.float64))
    cycle = _BlockVCycle(
        products=[laplacian_operator(matrix) for matrix, _, _ in levels],
        weights=[_inverse_absolute_row_sums(matrix)[:, None] for matrix, _, _ in levels],
        prolongators=[prolongator for _, prolongator, _ in levels],
        restrictions=[restriction for _, _, restriction in levels],
        coarsest=_deflated_inverse(coarsest_matrix.toarray(), candidate),
    )
    return LinearOperator(csr.shape, matvec=cycle, matmat=cycle, dtype=np.float64)


def _deflated_inverse(matrix, candidate):
    """The inverse of the dense ``matrix`` on the directions orthogonal to
    ``candidate``, its null vector.

    On the coarsest level the candidate is sqrt(degree)'s coarse image, whose
    eigenvalue is zero only up to rounding, so a plain pseudo-inverse could
    amplify that rounding enormously. Shifting the direction to 1 before
    inverting and projecting it out afterwards gives the inverse on the
    remaining directions exactly, however large that rounding is.
    """
    from scipy.linalg import pinvh

    unit = candidate / np.linalg.norm(candidate)
    outside = np.eye(len(unit)) - np.outer(unit, unit)
    return outside @ pinvh(matrix + np.outer(unit, unit)) @ outside


def _lobpcg_axes(csr, root_degree, dimensions, start, amg=False):
    """Return (axes, None), or (None, reason) when the result can't be trusted.

    With ``amg``, LOBPCG is preconditioned. LOBPCG overwrites ``start``.
    Nothing else refers to the preconditioner, so it and its hierarchy are
    freed as soon as this returns.
    """
    try:
        from scipy.sparse.linalg import lobpcg

        constraint = (root_degree / np.linalg.norm(root_degree)).reshape(-1, 1)
        operator = laplacian_operator(csr)
        with warnings.catch_warnings():
            # LOBPCG reports missed tolerances itself; the checks below decide.
            warnings.simplefilter("ignore")
            with Numba_Threads.limited_threads(Numba_Threads.default_thread_count()):
                preconditioner = amg_preconditioner(csr, root_degree) if amg else None
                values, vectors = lobpcg(
                    operator, start, M=preconditioner, Y=constraint, tol=LOBPCG_TOLERANCE,
                    maxiter=LOBPCG_MAX_ITERATIONS, largest=False,
                )
        order = np.argsort(values, kind="stable")[:dimensions]
        values, vectors = values[order], vectors[:, order]
    except Exception as error:
        return None, f"failed ({type(error).__name__}: {error})"
    if not (np.isfinite(values).all() and np.isfinite(vectors).all()):
        return None, "returned non-finite values"
    residual = max(
        float(np.linalg.norm(csr @ vectors[:, i] - values[i] * vectors[:, i])
              / np.linalg.norm(vectors[:, i]))
        for i in range(dimensions)
    )
    if not residual <= ACCEPTED_RESIDUAL:
        return None, f"did not converge (largest residual {residual:.2g})"
    if not float(np.max(_trivial_cosines(vectors, root_degree))) < ACCEPTED_TRIVIAL_COSINE:
        return None, "returned vectors that overlap sqrt(degree)"
    return vectors, None


def _arpack_axes(matrix, root_degree, dimensions, draw):
    from scipy.sparse.linalg import eigsh

    v0 = None if draw is None else draw(matrix.shape[0])
    values, vectors = eigsh(matrix, k=dimensions + 1, which="SM", tol=1e-3, v0=v0)
    return intended_axes(values, vectors, root_degree, dimensions)


def spectral_axes(
    local_edges,
    scores,
    node_count: int,
    dimensions: int,
    *,
    draw: Optional[Callable[..., np.ndarray]] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Return a component's (node_count, dimensions) start axes, lambda2 upward.

    ``draw(shape)`` supplies the job's random numbers, taken in a fixed order
    so a seeded start is reproducible: LOBPCG's start block, which the
    AMG-preconditioned solve uses first and plain LOBPCG reuses, then
    ARPACK's start vector if LOBPCG's result is rejected. Without it, LOBPCG
    draws from NumPy's global generator and ARPACK picks its own start vector.
    """
    laplacian, root_degree = normalized_laplacian(local_edges, scores, node_count)
    if node_count < LOBPCG_MIN_NODES:
        return _arpack_axes(laplacian, root_degree, dimensions, draw)
    csr = laplacian.tocsr()
    del laplacian
    start = (draw or np.random.standard_normal)((node_count, dimensions + LOBPCG_EXTRA_VECTORS))
    reason = None
    if amg_disabled():
        reason = f"disabled by {AMG_SWITCH_VARIABLE}"
    else:
        axes, failure = _lobpcg_axes(csr, root_degree, dimensions, start.copy(), amg=True)
        if axes is not None:
            if verbose:
                print(f"  > Spectral start ({node_count:,} nodes): LOBPCG with an AMG preconditioner.")
            return axes
        if verbose:
            print(f"  > LOBPCG with the AMG preconditioner {failure}; retrying without it.")
    axes, failure = _lobpcg_axes(csr, root_degree, dimensions, start)
    if axes is not None:
        if verbose:
            print(f"  > Spectral start ({node_count:,} nodes): LOBPCG without a preconditioner"
                  + (f" ({reason})." if reason else "."))
        return axes
    if verbose:
        print(f"  > LOBPCG spectral solve {failure}; using ARPACK instead.")
    return _arpack_axes(csr, root_degree, dimensions, draw)


def asinh_scaled(coordinates):
    """Spread one start axis: linear across its central 90%, logarithmic beyond.

    The axis is centred on its median and divided by half its 5-95% range
    before asinh, so outlying groups keep their order and side without
    squeezing the remaining nodes together. The result still goes through the
    usual min-max mapping into the component's box. Where the central range is
    (numerically) empty, as on stars, cliques and other symmetric components,
    half the full range is used instead.
    """
    low, middle, high = np.percentile(coordinates, [5.0, 50.0, 95.0])
    half_full = float(np.ptp(coordinates)) / 2.0
    spread = (high - low) / 2.0
    if not spread > half_full * EMPTY_CENTRE_SHARE:
        spread = half_full
    if not spread > 0.0:
        return np.asarray(coordinates, dtype=np.float64)
    return np.arcsinh((coordinates - middle) / spread)


__all__ = [
    "LOBPCG_MIN_NODES",
    "LOBPCG_EXTRA_VECTORS",
    "LOBPCG_TOLERANCE",
    "LOBPCG_MAX_ITERATIONS",
    "ACCEPTED_RESIDUAL",
    "ACCEPTED_TRIVIAL_COSINE",
    "TRIVIAL_COSINE",
    "EMPTY_CENTRE_SHARE",
    "AMG_SWITCH_VARIABLE",
    "AMG_MAX_COARSE",
    "AMG_COARSEST_LIMIT",
    "AMG_MAX_LEVELS",
    "NUMBA_AVAILABLE",
    "normalized_laplacian",
    "intended_axes",
    "laplacian_operator",
    "amg_disabled",
    "aggregation_levels",
    "amg_preconditioner",
    "spectral_axes",
    "asinh_scaled",
]
