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


def _cupy_to_device(array):
    """Copy an array to GPU memory."""
    return cp.asarray(array)


# Minimal backend registry. Each entry says whether the backend can be used, which
# array module implements it, and how to copy arrays to its memory and back to host
# memory. Other array libraries (e.g. JAX, PyTorch) can be added here without
# touching the functions below.
_BACKENDS = {
    "cpu": {
        "available": lambda: True,
        "get_module": lambda: np,
        "to_host": np.asarray,
        "to_device": np.asarray,
    },
    "cupy": {
        "available": _cupy_available,
        "get_module": lambda: cp,
        "to_host": _cupy_to_host,
        "to_device": _cupy_to_device,
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


def _get_backend(use_gpu=False):
    """Array module of the chosen backend, and functions copying to and from it.

    Parameters
    ----------
    use_gpu : bool, default False
        If True, use the GPU backend.

    Returns
    -------
    xp : module
        Array module (NumPy or CuPy)
    to_device : callable
        Copies an array to the memory of the backend
    to_host : callable
        Copies an array of the backend to host memory
    """
    backend = _BACKENDS[resolve_backend(use_gpu)]
    return backend["get_module"](), backend["to_device"], backend["to_host"]


# CUDA version of the loop of `hendrics.efsearch._fast_step` (`shift_and_sum`, then
# `_z_n_fast_cached`), one GPU thread per trial shift. The operations are done in the
# same order as in the Numba code, and multiplications and additions are not merged
# into one operation (``--fmad=false``, no fused multiply-add), so that the results
# are identical to the last bit. ``work`` holds one summed profile per trial.
_FAST_STEP_SOURCE = r"""
extern "C" __global__ void fast_step(
    const double* profiles, const double* lshift, const double* qshift,
    const double* base_shift, const double* quad_shift, const double* cached_cos,
    const double* cached_sin, int nprof, int nbin, int ntrial, int n,
    double* work, double* stats)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= ntrial) return;

    // shift_and_sum: add all sub-profiles, each shifted by a whole number of bins
    double* splat = work + (long long)t * nbin;
    for (int b = 0; b < nbin; b++) splat[b] = 0.0;
    for (int k = 0; k < nprof; k++) {
        double shift = rint(base_shift[k] * lshift[t] + quad_shift[k] * qshift[t]);
        // As np.mod, the result has the sign of the divisor
        double m = fmod(shift, (double)nbin);
        if (m < 0) m += nbin;
        int s = (int)m;
        const double* row = profiles + (long long)k * nbin;
        for (int b = 0; b < nbin; b++) {
            int src = b - s;
            if (src < 0) src += nbin;
            splat[b] += row[src];
        }
    }

    // _z_n_fast_cached
    double total = 0.0;
    for (int b = 0; b < nbin; b++) total += splat[b];
    double result = 0.0;
    for (int h = 1; h <= n; h++) {
        double sum_cos = 0.0, sum_sin = 0.0;
        for (int b = 0; b < nbin; b++) {
            sum_cos += cached_cos[b * h] * splat[b];
            sum_sin += cached_sin[b * h] * splat[b];
        }
        result += sum_cos * sum_cos + sum_sin * sum_sin;
    }
    stats[t] = 2.0 / total * result;
}
"""

_FAST_STEP_KERNEL = {}
_THREADS_PER_BLOCK = 256


def _fast_step_gpu(
    profiles, lshifts, qshifts, base_shift, quad_base_shift, cached_cos, cached_sin, n=1
):
    """Run the shift-and-sum and Z^2 of `hendrics.efsearch._fast_step` on the GPU.

    Parameters
    ----------
    profiles : `cupy.ndarray`
        Sub-profiles, of shape ``(nprof, nbin)``
    lshifts, qshifts : `cupy.ndarray`
        Linear and quadratic trial shifts, flattened (float64)
    base_shift, quad_base_shift, cached_cos, cached_sin : `cupy.ndarray`
        Output of `hendrics.efsearch._fast_step_constants`, copied to the GPU

    Other Parameters
    ----------------
    n : int, default 1
        Number of harmonics

    Returns
    -------
    stats : `cupy.ndarray`
        Z^2 statistics of each trial, in GPU memory
    """
    if "kernel" not in _FAST_STEP_KERNEL:
        _FAST_STEP_KERNEL["kernel"] = cp.RawKernel(
            _FAST_STEP_SOURCE, "fast_step", options=("--fmad=false",)
        )
    nprof, nbin = profiles.shape
    ntrial = lshifts.size
    stats = cp.empty(ntrial, dtype=cp.float64)
    work = cp.empty(ntrial * nbin, dtype=cp.float64)
    blocks = (ntrial + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _FAST_STEP_KERNEL["kernel"](
        (blocks,),
        (_THREADS_PER_BLOCK,),
        (
            cp.ascontiguousarray(profiles, dtype=cp.float64),
            lshifts,
            qshifts,
            base_shift,
            quad_base_shift,
            cached_cos,
            cached_sin,
            np.int32(nprof),
            np.int32(nbin),
            np.int32(ntrial),
            np.int32(n),
            work,
            stats,
        ),
    )
    return stats


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
