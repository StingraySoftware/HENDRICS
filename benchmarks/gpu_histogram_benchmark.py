"""Benchmark GPU histograms and averaged power spectra against the CPU versions.

Run manually on a machine with an NVIDIA GPU, CuPy and ``numba.cuda``, e.g.::

    python benchmarks/gpu_histogram_benchmark.py --sizes 100000 10000000

It compares:

1. 1D histograms: the HENDRICS Numba implementation on the CPU, a custom
   ``numba.cuda`` kernel (the prototype from PR #181, fixed) and ``cupy.histogram``
   (what :mod:`hendrics.gpu` uses).
2. 1D histogram followed by an FFT, all on the GPU.
3. 2D histograms with the shapes used by the ``--fast`` Z search: HENDRICS Numba
   implementation vs ``cupy.histogram2d``.
4. Averaged power spectra from events: stingray on the CPU; an "unfused" GPU loop,
   copying each light curve and each FFT back to host memory (what swapping
   stingray's ``histogram`` and ``fft`` with GPU versions would do); a "fused" GPU
   loop, where binning, FFT and averaging stay on the GPU with a single copy at
   the end. This measures whether keeping the data on the GPU matters.
5. One step of the ``--fast`` Z search, split into its parts. On the CPU: phases,
   2D histogram, and ``_fast_step`` (shifting and summing the sub-profiles, and
   Z^2). On the GPU: copying the events to the GPU (once per search), phases and
   sub-profiles, and ``_fast_step``. Run only this one with ``--benchmarks qffa``.

Each line reports the best time over ``--repeat`` calls (after a warm-up call that
also checks the result against the CPU version), and the speedup with respect to
the first line of the group.

Without CuPy, the CuPy parts are skipped. Without a GPU, the logic of the
``numba.cuda`` kernel can still be checked (but not timed meaningfully) with
numba's CUDA simulator, by setting ``NUMBA_ENABLE_CUDASIM=1``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import config, cuda
from stingray.fourier import avg_pds_from_timeseries, positive_fft_bins

from hendrics.base import hist1d_numba_seq, histogram2d
from hendrics.efsearch import (
    _fast_phase,
    _fast_phase_fddot,
    _fast_phase_fdot,
    _fast_step,
    _FastSearchOnDevice,
    _search_with_qffa_step,
)
from hendrics.gpu import HAS_CUPY, histogram2d_gpu, histogram_gpu

if HAS_CUPY:
    import cupy as cp

# Needed by some CUDA 12 installations to link the kernels
config.CUDA_ENABLE_PYNVJITLINK = 1

HAS_CUDA = cuda.is_available()
THREADS_PER_BLOCK = 256


@cuda.jit
def _hist1d_numba_gpu(H, tracks, bins, range_min, range_max):
    """CUDA kernel for computing a 1D histogram on the GPU.

    Parameters
    ----------
    H : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Device array representing the histogram bins to be filled
    tracks : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Device array of input values for which the histogram is computed
    bins : int
        Total number of histogram bins
    range_min : float
        Lower bound of the histogram range
    range_max : float
        Upper bound of the histogram range
    """
    delta = bins / (range_max - range_min)

    tid = cuda.grid(1)

    # Threads are launched in whole blocks, so some have tid >= tracks.size
    if tid < tracks.size:
        i = (tracks[tid] - range_min) * delta
        if 0 <= i < bins:
            cuda.atomic.add(H, int(i), 1)


def _launch_hist1d_kernel(d_hist, d_tracks, bins, ranges):
    blocks_per_grid = (d_tracks.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _hist1d_numba_gpu[blocks_per_grid, THREADS_PER_BLOCK](
        d_hist, d_tracks, bins, ranges[0], ranges[1]
    )


def hist1d_numba_gpu(a, bins, ranges):
    """Compute a 1D histogram using the custom CUDA kernel.

    Parameters
    ----------
    a : numpy.ndarray
        Input array of values to histogram
    bins : int
        Number of bins
    ranges : tuple of float
        The (min, max) range of values for the histogram. Values equal to the
        upper edge are excluded, as in :func:`hendrics.base.histogram`.

    Returns
    -------
    H : numpy.ndarray
        The histogram (float64), copied back to host memory
    """
    d_hist = cuda.to_device(np.zeros(bins, dtype=np.float64))
    _launch_hist1d_kernel(d_hist, cuda.to_device(np.asarray(a, dtype=np.float64)), bins, ranges)
    return d_hist.copy_to_host()


def hist1d_numba_gpu_fft(a, bins, ranges):
    """Compute a 1D histogram with the custom CUDA kernel, then its FFT with CuPy.

    The histogram is not copied to host memory between the two steps.

    Parameters
    ----------
    a : numpy.ndarray
        Input array of values to histogram
    bins : int
        Number of bins
    ranges : tuple of float
        The (min, max) range of values for the histogram

    Returns
    -------
    H_to_host : numpy.ndarray
        FFT of the histogram, copied back to host memory
    """
    d_hist = cp.zeros(bins, dtype=cp.float64)
    _launch_hist1d_kernel(d_hist, cp.asarray(a, dtype=cp.float64), bins, ranges)
    return cp.asnumpy(cp.fft.fft(d_hist))


def _to_host(array):
    if HAS_CUPY and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def _segment_starts_and_indices(times, gti, segment_size):
    """Start times of consecutive segments in the first GTI, and their event indices."""
    n_seg = int((gti[0, 1] - gti[0, 0]) // segment_size)
    starts = gti[0, 0] + np.arange(n_seg) * segment_size
    indices = np.searchsorted(times, np.append(starts, starts[-1] + segment_size))
    return starts, indices


def avg_pds_unfused(times, gti, segment_size, dt, xp):
    """Unnormalized averaged power spectrum, copying data to the host at each step.

    Parameters
    ----------
    times : numpy.ndarray
        Sorted event arrival times
    gti : numpy.ndarray
        Good time intervals; only the first one is used
    segment_size : float
        Length of each segment
    dt : float
        Bin time of the light curves
    xp : module
        Array module doing the binning and FFT (``cupy``, or ``numpy`` for checks)

    Returns
    -------
    power : numpy.ndarray
        Average of the squared FFT amplitudes, positive frequencies only
    """
    n_bin = int(np.rint(segment_size / dt))
    starts, indices = _segment_starts_and_indices(times, gti, segment_size)
    total = np.zeros(n_bin)
    for start, i0, i1 in zip(starts, indices[:-1], indices[1:], strict=True):
        segment = xp.asarray(times[i0:i1] - start)
        counts = _to_host(xp.histogram(segment, bins=n_bin, range=(0, segment_size))[0])
        ft = _to_host(xp.fft.fft(xp.asarray(counts, dtype=np.float64)))
        total += (ft * ft.conj()).real
    return total[positive_fft_bins(n_bin)] / starts.size


def avg_pds_fused(times, gti, segment_size, dt, xp):
    """Unnormalized averaged power spectrum, keeping all data on the device.

    Parameters
    ----------
    times : numpy.ndarray
        Sorted event arrival times
    gti : numpy.ndarray
        Good time intervals; only the first one is used
    segment_size : float
        Length of each segment
    dt : float
        Bin time of the light curves
    xp : module
        Array module doing the binning and FFT (``cupy``, or ``numpy`` for checks)

    Returns
    -------
    power : numpy.ndarray
        Average of the squared FFT amplitudes, positive frequencies only
    """
    n_bin = int(np.rint(segment_size / dt))
    starts, indices = _segment_starts_and_indices(times, gti, segment_size)
    d_times = xp.asarray(times)
    total = xp.zeros(n_bin, dtype=np.float64)
    for start, i0, i1 in zip(starts, indices[:-1], indices[1:], strict=True):
        counts = xp.histogram(d_times[i0:i1] - start, bins=n_bin, range=(0, segment_size))[0]
        ft = xp.fft.fft(counts.astype(np.float64))
        total += (ft * ft.conj()).real
    return _to_host(total[positive_fft_bins(n_bin)] / starts.size)


def _best_time(func, repeat):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - t0)
    return best


def _run(title, candidates, expected, repeat):
    print(title)
    reference = None
    for label, func in candidates.items():
        # The first call also warms up (JIT compilation, CUDA context, memory pools)
        agrees = np.allclose(func(), expected)
        elapsed = _best_time(func, repeat)
        reference = reference or elapsed
        print(
            f"  {label:<32} {elapsed * 1e3:10.2f} ms   x{reference / elapsed:7.2f}"
            f"   agrees with CPU: {agrees}"
        )


def benchmark_histogram(n_events, bins, repeat):
    """Time 1D histograms, with and without a following FFT."""
    x = np.random.default_rng(0).uniform(0, 1, n_events)
    ranges = [0.0, 1.0]

    candidates = {"numba CPU": lambda: hist1d_numba_seq(x, bins=bins, ranges=ranges)}
    if HAS_CUDA:
        candidates["numba.cuda kernel"] = lambda: hist1d_numba_gpu(x, bins, ranges)
    if HAS_CUPY:
        candidates["cupy.histogram"] = lambda: histogram_gpu(x, bins, ranges)
    expected = hist1d_numba_seq(x, bins=bins, ranges=ranges)
    _run(f"1D histogram, {n_events} events, {bins} bins", candidates, expected, repeat)

    if not HAS_CUPY:
        return
    candidates = {
        "numba CPU + numpy FFT": lambda: np.fft.fft(hist1d_numba_seq(x, bins=bins, ranges=ranges)),
        "cupy.histogram + cupy FFT": lambda: cp.asnumpy(
            cp.fft.fft(cp.histogram(cp.asarray(x), bins=bins, range=ranges)[0].astype(cp.float64))
        ),
    }
    if HAS_CUDA:
        candidates["numba.cuda kernel + cupy FFT"] = lambda: hist1d_numba_gpu_fft(x, bins, ranges)
    _run(f"1D histogram + FFT, {n_events} events", candidates, np.fft.fft(expected), repeat)


def benchmark_histogram2d(n_events, nbin, nprof, repeat):
    """Time 2D histograms with the shapes of the --fast Z search."""
    rng = np.random.default_rng(1)
    times = np.sort(rng.uniform(0, 1e5, n_events))
    phases = rng.uniform(0, 1, n_events)
    kwargs = {"bins": (nbin, nprof), "ranges": [[0, 1], [times[0], times[-1]]]}

    candidates = {"numba CPU": lambda: histogram2d(phases, times, **kwargs)}
    if HAS_CUPY:
        candidates["cupy.histogram2d"] = lambda: histogram2d_gpu(phases, times, **kwargs)
    expected = histogram2d(phases, times, **kwargs)
    _run(f"2D histogram, {n_events} events, {nbin}x{nprof} bins", candidates, expected, repeat)


def benchmark_avg_pds(n_events, segment_size, dt, n_segments, repeat):
    """Time averaged power spectra from events."""
    length = segment_size * n_segments
    times = np.sort(np.random.default_rng(2).uniform(0, length, n_events))
    gti = np.array([[0, length]])

    def stingray_cpu():
        # Returns an astropy Table
        pds = avg_pds_from_timeseries(times, gti, segment_size, dt, norm="none", silent=True)
        return pds["unnorm_power"]

    candidates = {"stingray CPU": stingray_cpu}
    if HAS_CUPY:
        candidates["CuPy, unfused"] = lambda: avg_pds_unfused(times, gti, segment_size, dt, cp)
        candidates["CuPy, fused"] = lambda: avg_pds_fused(times, gti, segment_size, dt, cp)
    n_bin = int(np.rint(segment_size / dt))
    _run(
        f"Averaged PDS, {n_events} events, {n_segments} segments x {n_bin} bins",
        candidates,
        stingray_cpu(),
        repeat,
    )


def _cpu_phases(times, mean_f, mean_fdot, mean_fddot):
    """Phases as in ``search_with_qffa_step``."""
    if mean_fddot != 0:
        return _fast_phase_fddot(times, mean_f, mean_fdot, mean_fddot)
    if mean_fdot != 0:
        return _fast_phase_fdot(times, mean_f, mean_fdot)
    return _fast_phase(times, mean_f)


def _sync():
    if HAS_CUPY:
        cp.cuda.Device().synchronize()


def benchmark_qffa_step(n_events, length, nbin, n, npfact, oversample, repeat):
    """Time the parts of one step of the --fast Z search, on CPU and GPU.

    Default parameters are those of ``HENzsearch --fast -N 2``. The phases use a
    frequency only, as that command does; the other phase formulas are only checked.
    """
    if nbin < 8 * n:
        raise ValueError("Z searches need at least 8 bins per harmonic")
    nprof = 8 * nbin * npfact
    rng = np.random.default_rng(3)
    # Centered, as in search_with_qffa
    times = np.sort(rng.uniform(-length / 2, length / 2, n_events))
    mean_f = 1.235
    hist_kwargs = {"range": [[0, 1], [times[0], times[-1]]], "bins": (nbin, nprof)}
    step_kwargs = {"nbin": nbin, "nprof": nprof, "npfact": npfact, "oversample": oversample}
    step_kwargs.update({"n": n, "length": length})

    # Trial shifts, as in search_with_qffa_step with search_fdot=True
    nshifts = max(int(np.rint(4 * oversample * npfact)), 1)
    shifts = np.linspace(-nbin * npfact, nbin * npfact, nshifts, endpoint=False)
    L, Q = np.meshgrid(shifts, shifts, indexing="ij")

    print(
        f"--fast search step, {n_events} events, {nprof} sub-profiles x {nbin} bins, "
        f"{L.size} trials"
    )

    def row(label, func):
        func()  # warm-up
        elapsed = _best_time(func, repeat)
        print(f"  {label:<52} {elapsed * 1e3:10.2f} ms")
        return elapsed

    def cpu_profiles(fdot=0, fddot=0):
        phases = _cpu_phases(times, mean_f, fdot, fddot)
        return np.ascontiguousarray(histogram2d(phases, times, **hist_kwargs).T)

    phases = _cpu_phases(times, mean_f, 0, 0)
    profiles = cpu_profiles()
    row("CPU: phases (_fast_phase)", lambda: _cpu_phases(times, mean_f, 0, 0))
    row("CPU: 2D histogram (numba)", lambda: histogram2d(phases, times, **hist_kwargs))
    row(
        "CPU: _fast_step (shift, sum, Z^2)",
        lambda: _fast_step(profiles, L, Q, shifts, shifts, nbin, n=n),
    )
    row("CPU: whole step", lambda: _search_with_qffa_step(times, mean_f, **step_kwargs))

    if not HAS_CUPY:
        return
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()

    def upload():
        result = _FastSearchOnDevice(times, nbin, nprof, use_gpu=True)
        _sync()
        return result

    row("GPU: copy events to the GPU (once per search)", upload)
    search = upload()

    def gpu_profiles():
        result = search.profiles(mean_f)
        _sync()
        return result

    d_profiles = gpu_profiles()
    row("GPU: phases + sub-profiles (bincount)", gpu_profiles)
    row(
        "GPU: _fast_step + copy back",
        lambda: search.stats(d_profiles, L, Q, shifts, shifts, n=n),
    )
    row(
        "GPU: whole step",
        lambda: _search_with_qffa_step(times, mean_f, device_search=search, **step_kwargs),
    )
    print(f"  GPU memory pool after the steps: {pool.total_bytes() / 2**20:.0f} MB")

    # Exact agreement with the CPU, for all phase formulas
    for fdot, fddot in [(0, 0), (-1e-11, 0), (-1e-11, 1e-17)]:
        same = np.array_equal(
            _to_host(search.profiles(mean_f, fdot, fddot)), cpu_profiles(fdot, fddot)
        )
        label = f"GPU sub-profiles, fdot={fdot:g}, fddot={fddot:g}"
        print(f"  {label:<52} agrees with CPU: {same}")
    expected = _search_with_qffa_step(times, mean_f, **step_kwargs)
    result = _search_with_qffa_step(times, mean_f, device_search=search, **step_kwargs)
    same = all(np.array_equal(exp, res) for exp, res in zip(expected, result, strict=True))
    print(f"  {'GPU whole step (grids and Z^2)':<52} agrees with CPU: {same}")

    search = d_profiles = None
    pool.free_all_blocks()
    print(f"  GPU memory pool after freeing: {pool.total_bytes() / 2**20:.0f} MB")


BENCHMARKS = ("histogram", "histogram2d", "pds", "qffa")


def main(args=None):
    """Run all benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark GPU vs CPU histograms and PDSs")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10**5, 10**6, 10**7], help="Numbers of events"
    )
    parser.add_argument("--bins", type=int, default=10**6, help="Bins of the 1D histograms")
    parser.add_argument("--nbin", type=int, default=16, help="Profile bins (2D histograms)")
    parser.add_argument("--nprof", type=int, default=256, help="Sub-profiles (2D histograms)")
    parser.add_argument("--segment-size", type=float, default=128, help="PDS segment length")
    parser.add_argument("--dt", type=float, default=1 / 4096, help="PDS bin time")
    parser.add_argument("--n-segments", type=int, default=64, help="Number of PDS segments")
    parser.add_argument("--qffa-nbin", type=int, default=128, help="Profile bins (--fast step)")
    parser.add_argument("--qffa-n", type=int, default=2, help="Harmonics (--fast step)")
    parser.add_argument("--qffa-npfact", type=int, default=2, help="npfact (--fast step)")
    parser.add_argument("--qffa-oversample", type=float, default=4, help="Oversampling (--fast)")
    parser.add_argument("--qffa-length", type=float, default=1e5, help="Obs. length (--fast)")
    parser.add_argument(
        "--benchmarks", nargs="+", choices=BENCHMARKS, default=BENCHMARKS, help="What to run"
    )
    parser.add_argument("--repeat", type=int, default=3, help="Timed calls per candidate")
    args = parser.parse_args(args)

    if not HAS_CUPY:
        print("CuPy is not installed: skipping the CuPy benchmarks")
    if not HAS_CUDA:
        print("No CUDA device: skipping the numba.cuda kernel")

    for n_events in args.sizes:
        if "histogram" in args.benchmarks:
            benchmark_histogram(n_events, args.bins, args.repeat)
        if "histogram2d" in args.benchmarks:
            benchmark_histogram2d(n_events, args.nbin, args.nprof, args.repeat)
        if "pds" in args.benchmarks:
            benchmark_avg_pds(n_events, args.segment_size, args.dt, args.n_segments, args.repeat)
        if "qffa" in args.benchmarks:
            benchmark_qffa_step(
                n_events,
                args.qffa_length,
                args.qffa_nbin,
                args.qffa_n,
                args.qffa_npfact,
                args.qffa_oversample,
                args.repeat,
            )


if __name__ == "__main__":
    main()
