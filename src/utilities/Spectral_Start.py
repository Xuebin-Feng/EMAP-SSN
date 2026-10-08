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
Numba threads. A result that misses the residual or orthogonality check falls
back to ARPACK, and smaller components keep ARPACK. ``asinh_scaled`` then
spreads each axis, so a few outlying nodes no longer squeeze the rest of a
component into a dot.
"""

from __future__ import annotations

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


def _lobpcg_axes(csr, root_degree, dimensions, draw):
    """Return (axes, None), or (None, reason) when the result can't be trusted."""
    try:
        from scipy.sparse.linalg import lobpcg

        start = draw((csr.shape[0], dimensions + LOBPCG_EXTRA_VECTORS))
        constraint = (root_degree / np.linalg.norm(root_degree)).reshape(-1, 1)
        operator = laplacian_operator(csr)
        with warnings.catch_warnings():
            # LOBPCG reports missed tolerances itself; the checks below decide.
            warnings.simplefilter("ignore")
            with Numba_Threads.limited_threads(Numba_Threads.default_thread_count()):
                values, vectors = lobpcg(
                    operator, start, Y=constraint, tol=LOBPCG_TOLERANCE,
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
    so a seeded start is reproducible: LOBPCG's start block, then ARPACK's
    start vector if LOBPCG's result is rejected. Without it, LOBPCG draws from
    NumPy's global generator and ARPACK picks its own start vector.
    """
    laplacian, root_degree = normalized_laplacian(local_edges, scores, node_count)
    if node_count < LOBPCG_MIN_NODES:
        return _arpack_axes(laplacian, root_degree, dimensions, draw)
    csr = laplacian.tocsr()
    del laplacian
    axes, reason = _lobpcg_axes(csr, root_degree, dimensions, draw or np.random.standard_normal)
    if axes is not None:
        return axes
    if verbose:
        print(f"  > LOBPCG spectral solve {reason}; using ARPACK instead.")
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
    "NUMBA_AVAILABLE",
    "normalized_laplacian",
    "intended_axes",
    "laplacian_operator",
    "spectral_axes",
    "asinh_scaled",
]
