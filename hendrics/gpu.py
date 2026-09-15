"""Optional GPU implementations of the HENDRICS histograms.

Nothing in this module requires a GPU: CuPy is only imported if installed, the
presence of a CUDA device is checked lazily, and the GPU code only runs when
explicitly requested.

The GPU functions are drop-in replacements for :func:`hendrics.base.histogram`
and :func:`hendrics.base.histogram2d`: same bin-edge convention (values equal to
the upper edge of the range are excluded, unlike :func:`numpy.histogram`) and
same output dtypes.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

__all__ = ["HAS_CUPY", "histogram2d_gpu", "histogram_gpu", "resolve_backend"]


def _cupy_available():
    """Return True if CuPy is installed and at least one CUDA device is visible."""
    if not HAS_CUPY:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    # CuPy raises different exceptions for a missing driver or missing libraries
    except Exception:
        return False


def _cupy_to_host(array):
    """Copy a CuPy array to host memory."""
    return cp.asnumpy(array)


# Minimal backend registry. Each entry says whether the backend can be used, which
# array module implements it, and how to copy its arrays back to host memory.
# Other array libraries (e.g. JAX, PyTorch) can be added here without touching
# the functions below.
_BACKENDS = {
    "cpu": {
        "available": lambda: True,
        "get_module": lambda: np,
        "to_host": np.asarray,
    },
    "cupy": {
        "available": _cupy_available,
        "get_module": lambda: cp,
        "to_host": _cupy_to_host,
    },
}


def resolve_backend(use_gpu=False):
    """Choose the array backend.

    Parameters
    ----------
    use_gpu : bool, default False
        If True, request the GPU backend.

    Returns
    -------
    backend : str
        ``"cupy"`` if ``use_gpu`` is True, ``"cpu"`` otherwise.

    Raises
    ------
    RuntimeError
        If the GPU is requested but CuPy or a CUDA device is not available.
    """
    if not use_gpu:
        return "cpu"
    if not _BACKENDS["cupy"]["available"]():
        msg = (
            "use_gpu=True requires CuPy and a visible CUDA device. Install CuPy "
            "(e.g. `pip install cupy-cuda12x`) or set use_gpu=False."
        )
        raise RuntimeError(msg)
    return "cupy"


def _gpu_backend():
    backend = _BACKENDS[resolve_backend(use_gpu=True)]
    return backend["get_module"](), backend["to_host"]


def histogram_gpu(a, bins, ranges, weights=None):
    """Compute a 1D histogram on the GPU.

    Parameters
    ----------
    a : array-like
        Input values.
    bins : int
        Number of bins.
    ranges : (float, float)
        Lower and upper edge of the histogram. Values equal to the upper edge
        are excluded, as in :func:`hendrics.base.histogram`.

    Other Parameters
    ----------------
    weights : array-like, optional
        Weights of each value.

    Returns
    -------
    hist : `np.ndarray`
        Histogram (float64), copied back to host memory.
    """
    xp, to_host = _gpu_backend()
    lo, hi = ranges
    a = xp.asarray(a, dtype=np.float64)
    good = a < hi
    if weights is not None:
        weights = xp.asarray(weights, dtype=np.float64)[good]

    hist = xp.histogram(a[good], bins=bins, range=(lo, hi), weights=weights)[0]
    return to_host(hist.astype(np.float64))


def histogram2d_gpu(x, y, bins, ranges, weights=None):
    """Compute a 2D histogram on the GPU.

    Parameters
    ----------
    x : array-like
        Values along the first axis.
    y : array-like
        Values along the second axis.
    bins : (int, int)
        Number of bins along each axis.
    ranges : [[float, float], [float, float]]
        Lower and upper edges along each axis. Values equal to an upper edge
        are excluded, as in :func:`hendrics.base.histogram2d`.

    Other Parameters
    ----------------
    weights : array-like, optional
        Weights of each value.

    Returns
    -------
    hist : `np.ndarray`
        Histogram, copied back to host memory. Unsigned 64-bit integers if
        unweighted, float64 if weighted, as in :func:`hendrics.base.histogram2d`.
    """
    xp, to_host = _gpu_backend()
    (xlo, xhi), (ylo, yhi) = ranges
    x = xp.asarray(x, dtype=np.float64)
    y = xp.asarray(y, dtype=np.float64)
    good = (x < xhi) & (y < yhi)

    dtype = np.uint64
    if weights is not None:
        weights = xp.asarray(weights, dtype=np.float64)[good]
        dtype = np.float64

    hist = xp.histogram2d(
        x[good], y[good], bins=bins, range=[[xlo, xhi], [ylo, yhi]], weights=weights
    )[0]
    return to_host(hist.astype(dtype))
