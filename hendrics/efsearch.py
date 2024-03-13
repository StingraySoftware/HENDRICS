# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Search for pulsars."""

import warnings
import os
import argparse
import copy

import numpy as np
from scipy.ndimage import gaussian_filter
from astropy import log
from astropy.table import Table
from astropy.logger import AstropyUserWarning
from .io import get_file_type
from stingray.pulse.search import (
    epoch_folding_search,
    z_n_search,
    search_best_peaks,
)
from stingray.stats import a_from_ssig, pf_from_ssig, power_confidence_limits

from stingray.gti import time_intervals_from_gtis
from stingray.utils import assign_value_if_none
from stingray.pulse.modeling import fit_sinc, fit_gaussian
from stingray.stats import pf_upper_limit
from .io import (
    load_events,
    EFPeriodogram,
    save_folding,
    HEN_FILE_EXTENSION,
    load_folding,
)

from .base import (
    hen_root,
    show_progress,
    adjust_dt_for_power_of_two,
    HENDRICS_STAR_VALUE,
)
from .base import deorbit_events, njit, prange, vectorize, float64
from .base import histogram2d, histogram, memmapped_arange
from .base import z2_n_detection_level, fold_detection_level
from .base import find_peaks_in_image
from .base import z2_n_detection_level
from .base import fold_detection_level
from .fold import filter_energy
from .ffa import _z_n_fast_cached, ffa_search, h_test
from .fake import scramble

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


D_OMEGA_FACTOR = 2 * np.sqrt(3)
TWOPI = 2 * np.pi


__all__ = [
    "check_phase_error_after_casting_to_double",
    "decide_binary_parameters",
    "folding_orbital_search",
    "fit",
    "calculate_shifts",
    "mod",
    "shift_and_sum",
    "z_n_fast",
    "transient_search",
    "plot_transient_search",
    "search_with_qffa_step",
    "search_with_qffa",
    "search_with_ffa",
    "folding_search",
    "dyn_folding_search",
    "main_efsearch",
    "main_zsearch",
    "z2_vs_pf",
    "main_z2vspf",
    "main_accelsearch",
    "h_test",
]


def find_nearest_contour(cs, x, y, indices=None, pixel=True):
    """
    Find the point in the contour plot that is closest to ``(x, y)``.

    This method does not support filled contours.

    ..note::

        This function was deprecated from matplotlib, and we copied it here to
        ensure code stability

    Parameters
    ----------
    x, y : float
        The reference point.
    indices : list of int or None, default: None
        Indices of contour levels to consider.  If None (the default), all
        levels are considered.
    pixel : bool, default: True
        If *True*, measure distance in pixel (screen) space, which is
        useful for manual contour labeling; else, measure distance in axes
        space.

    Returns
    -------
    contour : `.Collection`
        The contour that is closest to ``(x, y)``.
    segment : int
        The index of the `.Path` in *contour* that is closest to
        ``(x, y)``.
    index : int
        The index of the path segment in *segment* that is closest to
        ``(x, y)``.
    xmin, ymin : float
        The point in the contour plot that is closest to ``(x, y)``.
    d2 : float
        The squared distance from ``(xmin, ymin)`` to ``(x, y)``.
    """

    from matplotlib.contour import _find_closest_point_on_path

    # This function uses a method that is probably quite
    # inefficient based on converting each contour segment to
    # pixel coordinates and then comparing the given point to
    # those coordinates for each contour.  This will probably be
    # quite slow for complex contours, but for normal use it works
    # sufficiently well that the time is not noticeable.
    # Nonetheless, improvements could probably be made.

    if cs.filled:
        raise ValueError("Method does not support filled contours.")

    if indices is None:
        indices = range(len(cs.collections))

    d2min = np.inf
    conmin = None
    segmin = None
    imin = None
    xmin = None
    ymin = None

    point = np.array([x, y])

    for icon in indices:
        con = cs.collections[icon]
        trans = con.get_transform()
        paths = con.get_paths()

        for segNum, linepath in enumerate(paths):
            lc = linepath.vertices
            # transfer all data points to screen coordinates if desired
            if pixel:
                lc = trans.transform(lc)

            d2, xc, leg = _find_closest_point_on_path(lc, point)
            if d2 < d2min:
                d2min = d2
                conmin = icon
                segmin = segNum
                imin = leg[1]
                xmin = xc[0]
                ymin = xc[1]

    return (conmin, segmin, imin, xmin, ymin, d2min)


def _save_df_to_csv(df, csv_file, reset=False):
    if not os.path.exists(csv_file) or reset:
        mode = "w"
        header = True
    else:
        mode = "a"
        header = False
    df.to_csv(csv_file, header=header, index=False, mode=mode)


def check_phase_error_after_casting_to_double(tref, f, fdot=0):
    """Check the maximum error expected in the phase when casting to double."""
    times = np.array(np.random.normal(tref, 0.1, 1000), dtype=np.longdouble)
    times_dbl = times.astype(np.double)
    phase = times * f + 0.5 * times**2 * fdot
    phase_dbl = times_dbl * np.double(f) + 0.5 * times_dbl**2 * np.double(fdot)
    return np.max(np.abs(phase_dbl - phase))


def decide_binary_parameters(
    length,
    freq_range,
    porb_range,
    asini_range,
    fdot_range=[0, 0],
    NMAX=10,
    csv_file="db.csv",
    reset=False,
):
    import pandas as pd

    count = 0
    omega_range = [1 / porb_range[1], 1 / porb_range[0]]
    columns = [
        "freq",
        "fdot",
        "X",
        "Porb",
        "done",
        "max_stat",
        "min_stat",
        "best_T0",
    ]

    df = 1 / length
    log.info(
        "Recommended frequency steps: {}".format(int(np.diff(freq_range)[0] // df + 1))
    )
    while count < NMAX:
        # In any case, only the first loop deletes the file
        if count > 0:
            reset = False
        block_of_data = []
        freq = np.random.uniform(freq_range[0], freq_range[1])
        fdot = np.random.uniform(fdot_range[0], fdot_range[1])

        dX = 1 / (TWOPI * freq)

        nX = int(np.diff(asini_range)[0] // dX) + 1
        Xs = np.random.uniform(asini_range[0], asini_range[1], nX)

        for X in Xs:
            dOmega = 1 / (TWOPI * freq * X * length) * D_OMEGA_FACTOR
            nOmega = int(np.diff(omega_range)[0] // dOmega) + 1
            Omegas = np.random.uniform(omega_range[0], omega_range[1], nOmega)

            for Omega in Omegas:
                block_of_data.append(
                    [freq, fdot, X, TWOPI / Omega, False, 0.0, 0.0, 0.0]
                )

        df = pd.DataFrame(block_of_data, columns=columns)
        _save_df_to_csv(df, csv_file, reset=reset)
        count += 1

    return csv_file


def folding_orbital_search(
    events,
    parameter_csv_file,
    chunksize=100,
    outfile="out.csv",
    fun=epoch_folding_search,
    **fun_kwargs,
):
    import pandas as pd

    times = (events.time - events.gti[0, 0]).astype(np.float64)
    for chunk in pd.read_csv(parameter_csv_file, chunksize=chunksize):
        try:
            chunk["done"][0]
        except Exception:
            continue
        for i in range(len(chunk)):
            if chunk["done"][i]:
                continue

            row = chunk.iloc[i]
            freq, fdot, X, Porb = np.array(
                [row["freq"], row["fdot"], row["X"], row["Porb"]],
                dtype=np.float64,
            )

            dT0 = min(1 / (TWOPI**2 * freq) * Porb / X, Porb / 10)
            max_stats = 0
            min_stats = 1e32
            best_T0 = None
            T0s = np.random.uniform(0, Porb, int(Porb // dT0 + 1))
            for T0 in T0s:
                # one iteration
                new_values = times - X * np.sin(2 * np.pi * (times - T0) / Porb)
                new_values = new_values - X * np.sin(
                    2 * np.pi * (new_values - T0) / Porb
                )
                fgrid, stats = fun(
                    new_values, np.array([freq]), fdots=fdot, **fun_kwargs
                )
                if stats[0] > max_stats:
                    max_stats = stats[0]
                    best_T0 = T0
                if stats[0] < min_stats:
                    min_stats = stats[0]
            idx = chunk.index[i]
            chunk.iloc[idx, chunk.columns.get_loc("max_stat")] = max_stats
            chunk.iloc[idx, chunk.columns.get_loc("min_stat")] = min_stats
            chunk.iloc[idx, chunk.columns.get_loc("best_T0")] = best_T0

            chunk.iloc[idx, chunk.columns.get_loc("done")] = True
        _save_df_to_csv(chunk, outfile)


def fit(frequencies, stats, center_freq, width=None, obs_length=None, baseline=0):
    estimated_amp = stats[np.argmin(np.abs(frequencies - center_freq))]

    if obs_length is not None:
        s = fit_sinc(
            frequencies,
            stats - baseline,
            obs_length=obs_length,
            amp=estimated_amp,
            mean=center_freq,
        )
    else:
        df = frequencies[1] - frequencies[0]
        if width is None:
            width = 2 * df
        s = fit_gaussian(
            frequencies,
            stats - baseline,
            stddev=width,
            amplitude=estimated_amp,
            mean=center_freq,
        )

    return s


@njit()
def calculate_shifts(nprof: int, nbin: int, nshift: int, order: int = 1) -> np.array:
    shifts = np.linspace(-1.0, 1.0, nprof) ** order
    return nshift * shifts


@njit()
def mod(num, n2):
    return np.mod(num, n2)


@njit()
def shift_and_sum(
    repeated_profiles, lshift, qshift, splat_prof, base_shift, quadbaseshift
):
    nprof = repeated_profiles.shape[0]
    nbin = splat_prof.size
    twonbin = nbin * 2
    splat_prof[:] = 0.0
    for k in range(nprof):
        total_shift = base_shift[k] * lshift + quadbaseshift[k] * qshift
        total_shift = mod(np.rint(total_shift), nbin)
        total_shift_int = int(total_shift)

        splat_prof[:] += repeated_profiles[
            k, nbin - total_shift_int : twonbin - total_shift_int
        ]

    return splat_prof


@njit(fastmath=True)
def z_n_fast(phase, norm, n=2):
    """Z^2_n statistics, a` la Buccheri+03, A&A, 128, 245, eq. 2.

    Here in a fast implementation based on numba.
    Assumes that nbin != 0 and norm is an array.

    Parameters
    ----------
    phase : array of floats
        The phases of the events, in terms of 2PI
    norm : float or array of floats
        A normalization factor that gets multiplied as a weight.
    n : int, default 2
        The ``n`` in $Z^2_n$.

    Returns
    -------
    z2_n : float
        The Z^2_n statistics of the events.

    Examples
    --------
    >>> phase = 2 * np.pi * np.arange(0, 1, 0.01)
    >>> norm = np.sin(phase) + 1
    >>> assert np.isclose(z_n_fast(phase, norm, n=4), 50)
    >>> assert np.isclose(z_n_fast(phase, norm, n=2), 50)
    """

    total_norm = np.sum(norm)

    result = 0
    # Instead of calculating k phi each time
    kph = np.zeros_like(phase)

    for k in range(1, n + 1):
        kph += phase
        result += np.sum(np.cos(kph) * norm) ** 2 + np.sum(np.sin(kph) * norm) ** 2

    return 2 / total_norm * result


@njit()
def _average_and_z_sub_search(profiles, n=2):
    """Z^2_n statistics calculated in sub-profiles.

    Parameters
    ----------
    profiles : array of arrays
        a M x N matrix containing a list of pulse profiles
    nbin : int
        The number of bins in the profiles.
    Returns
    -------
    z2_n : float array (MxM)
        The Z^2_n statistics of the events.

    Examples
    --------
    >>> phase = 2 * np.pi * np.arange(0, 1, 0.01)
    >>> norm = np.sin(phase) + 1
    >>> profiles = np.ones((16, len(norm)))
    >>> profiles[8] = norm
    >>> n_ave, results = _average_and_z_sub_search(profiles, n=2)
    >>> assert np.isclose(results[0, 8], 50)
    >>> assert np.isclose(results[1, 8], 50/2)
    >>> assert np.isclose(results[2, 8], 50/4)
    >>> assert np.isclose(results[3, 8], 50/8)
    """
    nprof = len(profiles)
    # Only use powers of two
    nprof = int(2 ** np.log2(nprof))
    profiles = profiles[:nprof]

    nbin = len(profiles[0])

    n_log_ave_max = int(np.log2(nprof))

    results = np.zeros((n_log_ave_max, nprof))

    twopiphases = 2 * np.pi * np.arange(0, 1, 1 / nbin)

    n_ave = 2 ** np.arange(n_log_ave_max)

    for ave_i in range(len(n_ave)):
        n_ave_i = n_ave[ave_i]
        shape_0 = int(profiles.shape[0] / n_ave_i)
        # new_profiles = np.zeros((shape_0, profiles.shape[1]))
        for i in range(shape_0):
            new_profiles = np.sum(profiles[i * n_ave_i : (i + 1) * n_ave_i], axis=0)

            # Work around strange numba bug. Will reinstate np.max when it's
            # solved
            if np.sum(new_profiles) == 0:
                continue

            z = z_n_fast(twopiphases, norm=new_profiles, n=n)
            results[ave_i, i * n_ave_i : (i + 1) * n_ave_i] = z

    return n_ave, results


def _transient_search_step(
    times: np.double, mean_f: np.double, mean_fdot=0, nbin=16, nprof=64, n=1
):
    """Single step of transient search."""

    # Cast to standard double, or Numba's histogram2d will fail
    # horribly.

    phases = _fast_phase_fdot(times, mean_f, mean_fdot)

    profiles = histogram2d(
        phases,
        times,
        range=[[0, 1], [times[0], times[-1]]],
        bins=(nbin, nprof),
    ).T

    n_ave, results = _average_and_z_sub_search(profiles, n=n)
    return n_ave, results


class TransientResults(object):
    oversample: int = None
    f0: float = None
    f1: float = None
    fdot: float = None
    nave: int = None
    freqs: np.array = None
    times: np.array = None
    stats: np.array = None


def transient_search(
    times,
    f0,
    f1,
    fdot=0,
    nbin=16,
    nprof=None,
    n=1,
    t0=None,
    t1=None,
    oversample=4,
):
    """Search for transient pulsations.

    Parameters
    ----------
    times : array of floats
        Arrival times of photons
    f0 : float
        Minimum frequency to search
    f1 : float
        Maximum frequency to search

    Other parameters
    ----------------
    nbin : int
        Number of bins to divide the profile into
    nprof : int, default None
        number of slices of the dataset to use. If None, we use 8 times nbin.
        Motivation in the comments.
    npfact : int, default 2
        maximum "sliding" of the dataset, in phase.
    oversample : int, default 8
        Oversampling wrt the standard FFT delta f = 1/T
    search_fdot : bool, default False
        Switch fdot search on or off
    t0 : float, default min(times)
        starting time
    t1 : float, default max(times)
        stop time
    """
    if nprof is None:
        # total_delta_phi = 2 == dnu * T
        # In a single sub interval
        # delta_phi = dnu * t
        # with t = T / nprof
        # so dnu T / nprof < 1 / nbin, and
        # nprof > total_delta_phi * nbin to get all the signal inside one bin
        # in a given sub-integration
        nprof = 4 * 2 * nbin

    times = copy.deepcopy(times)

    if t0 is None:
        t0 = times.min()
    if t1 is None:
        t1 = times.max()
    meantime = (t1 + t0) / 2
    times -= meantime

    maxerr = check_phase_error_after_casting_to_double(np.max(times), f1, fdot)
    log.info(
        f"Maximum error on the phase expected when casting to double: " f"{maxerr}"
    )
    if maxerr > 1 / nbin / 10:
        warnings.warn(
            "Casting to double produces non-negligible phase errors. "
            "Please use shorter light curves.",
            AstropyUserWarning,
        )

    times = times.astype(np.double)

    length = t1 - t0

    frequency = (f0 + f1) / 2

    # Step: npfact * 1 / T

    step = 1 / length / oversample

    niter = int(np.rint((f1 - f0) / step)) + 2

    allvalues = list(range(-(niter // 2), niter // 2))
    if allvalues == []:
        allvalues = [0]

    all_results = []
    all_freqs = []

    dt = (times[-1] - times[0]) / nprof

    for ii, i in enumerate(show_progress(allvalues)):
        offset = step * i
        fdot_offset = 0

        mean_f = np.double(frequency + offset + 0.12 * step)
        mean_fdot = np.double(fdot + fdot_offset)
        nave, results = _transient_search_step(
            times, mean_f, mean_fdot=mean_fdot, nbin=nbin, nprof=nprof, n=n
        )
        all_results.append(results)
        all_freqs.append(mean_f)

    all_results = np.array(all_results)
    all_freqs = np.array(all_freqs)

    times = dt * np.arange(all_results.shape[2])

    results = TransientResults()
    results.oversample = oversample
    results.f0 = f0
    results.f1 = f1
    results.fdot = fdot
    results.nave = nave
    results.freqs = all_freqs
    results.times = times
    results.stats = np.array([all_results[:, i, :].T for i in range(nave.size)])

    return results


def plot_transient_search(results, gif_name=None):
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")
    if gif_name is None:
        gif_name = "transients.gif"

    all_images = []
    for i, (ima, nave) in enumerate(zip(results.stats, results.nave)):
        f = results.freqs
        t = results.times
        nprof = ima.shape[0]
        oversample = results.oversample

        # To calculate ntrial, we need to take into account that
        # 1. the image has nave equal pixels
        # 2. the frequency axis is oversampled by at least nprof / nave
        ntrial = max(int(ima.size / nave / (nprof / nave) / oversample), 1)

        detl = z2_n_detection_level(epsilon=0.0015, n=2, ntrial=ntrial)

        # To calculate ntrial from the summed image, we use the
        # length of the frequency axis, considering oversample by
        # nprof / nave:
        ntrial_sum = max(int(f.size / nave / (nprof / nave) / oversample), 1)

        sum_detl = z2_n_detection_level(
            epsilon=0.0015,
            n=2,
            ntrial=ntrial_sum,
            n_summed_spectra=nprof / nave,
        )
        fig = plt.figure(figsize=(10, 10))
        plt.clf()
        gs = plt.GridSpec(2, 2, height_ratios=(1, 3))
        for i_f in [0, 1]:
            axf = plt.subplot(gs[0, i_f])
            axima = plt.subplot(gs[1, i_f], sharex=axf)

            axima.pcolormesh(f, t, ima / detl * 3, vmax=3, vmin=0.3, shading="nearest")

            mean_line = np.mean(ima, axis=0) / sum_detl * 3
            maxidx = np.argmax(mean_line)
            maxline = mean_line[maxidx]
            best_f = f[maxidx]
            for il, line in enumerate(ima / detl * 3):
                axf.plot(
                    f,
                    line,
                    lw=0.2,
                    ls="-",
                    c="grey",
                    alpha=0.5,
                    label=f"{il}",
                )
                maxidx = np.argmax(mean_line)
                if line[maxidx] > maxline:
                    best_f = f[maxidx]
                    maxline = line[maxidx]
            if 3.5 < maxline < 5 and i_f == 0:  # pragma: no cover
                print(
                    f"{gif_name}: Possible candidate at step {i}: {best_f} Hz (~{maxline:.1f} sigma)"
                )
            elif maxline >= 5 and i_f == 0:  # pragma: no cover
                print(
                    f"{gif_name}: Candidate at step {i}: {best_f} Hz (~{maxline:.1f} sigma)"
                )

            axf.plot(f, mean_line, lw=1, c="k", zorder=10, label="mean", ls="-")

            axima.set_xlabel("Frequency")
            axima.set_ylabel("Time")
            axf.set_ylabel(r"Significance ($\sigma$)")
            nhigh = len(t)
            df = (f[1] - f[0]) * oversample * nhigh
            xmin = max(best_f - df, results.f0)
            xmax = min(best_f + df, results.f1)
            if i_f == 0:
                axf.set_xlim([results.f0, results.f1])
                axf.axvline(xmin, ls="--", c="b", lw=2)
                axf.axvline(xmax, ls="--", c="b", lw=2)
            else:
                axf.set_xlim([xmin, xmax])

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba().cast("B"), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        plt.close(fig)
        all_images.append(image)

    if HAS_IMAGEIO:
        imageio.v3.imwrite(gif_name, all_images, duration=1000.0)
    else:
        warnings.warn(
            "imageio needed to save the transient search results " "into a gif image."
        )

    return all_images


@njit(nogil=True, parallel=True)
def _fast_step(profiles, L, Q, linbinshifts, quabinshifts, nbin, n=2):
    twopiphases = 2 * np.pi * np.arange(0, 1, 1 / nbin)

    cached_cos = np.zeros(n * nbin)
    cached_sin = np.zeros(n * nbin)
    for i in range(n):
        cached_cos[i * nbin : (i + 1) * nbin] = np.cos(twopiphases)
        cached_sin[i * nbin : (i + 1) * nbin] = np.sin(twopiphases)

    stats = np.zeros_like(L)
    repeated_profiles = np.hstack((profiles, profiles, profiles))

    nprof = repeated_profiles.shape[0]

    base_shift = np.linspace(-1, 1, nprof)
    quad_base_shift = base_shift**2

    for i in prange(linbinshifts.size):
        # This zeros needs to be here, not outside the parallel loop, or
        # the threads will try to write it all at the same time
        splat_prof = np.zeros(nbin)
        for j in range(quabinshifts.size):
            splat_prof = shift_and_sum(
                repeated_profiles,
                L[i, j],
                Q[i, j],
                splat_prof,
                base_shift,
                quad_base_shift,
            )
            local_stat = _z_n_fast_cached(splat_prof, cached_cos, cached_sin, n=n)
            stats[i, j] = local_stat

    return stats


@njit(parallel=True)
def _fast_phase_fdot(ts, mean_f, mean_fdot=0):
    phases = ts * mean_f + 0.5 * ts * ts * mean_fdot
    return phases - np.floor(phases)


ONE_SIXTH = 1 / 6


@njit(parallel=True)
def _fast_phase_fddot(ts, mean_f, mean_fdot=0, mean_fddot=0):
    tssq = ts * ts
    phases = ts * mean_f + 0.5 * tssq * mean_fdot + ONE_SIXTH * tssq * ts * mean_fddot
    return phases - np.floor(phases)


@njit(parallel=True)
def _fast_phase(ts, mean_f):
    phases = ts * mean_f
    return phases - np.floor(phases)


def search_with_qffa_step(
    times: np.double,
    mean_f: np.double,
    mean_fdot=0,
    mean_fddot=0,
    nbin=16,
    nprof=64,
    npfact=2,
    oversample=8,
    n=1,
    search_fdot=True,
):
    """Single step of quasi-fast folding algorithm."""
    # Cast to standard double, or Numba's histogram2d will fail
    # horribly.

    if mean_fddot != 0:
        phases = _fast_phase_fddot(times, mean_f, mean_fdot, mean_fddot)
    elif mean_fdot != 0:
        phases = _fast_phase_fdot(times, mean_f, mean_fdot)
    else:
        phases = _fast_phase(times, mean_f)

    profiles = histogram2d(
        phases,
        times,
        range=[[0, 1], [times[0], times[-1]]],
        bins=(nbin, nprof),
    ).T

    # Assume times are sorted
    t1, t0 = times[-1], times[0]

    # dn = max(1, int(nbin / oversample))
    linbinshifts = np.linspace(-nbin * npfact, nbin * npfact, int(oversample * npfact))
    if search_fdot:
        quabinshifts = np.linspace(
            -nbin * npfact, nbin * npfact, int(oversample * npfact)
        )
    else:
        quabinshifts = np.array([0])

    dphi = 1 / nbin
    delta_t = (t1 - t0) / 2
    bin_to_frequency = dphi / delta_t
    bin_to_fdot = 2 * dphi / delta_t**2

    L, Q = np.meshgrid(linbinshifts, quabinshifts, indexing="ij")

    stats = _fast_step(profiles, L, Q, linbinshifts, quabinshifts, nbin, n=n)

    return L * bin_to_frequency + mean_f, Q * bin_to_fdot + mean_fdot, stats


def search_with_qffa(
    times,
    f0,
    f1,
    fdot=0,
    fddot=0,
    nbin=16,
    nprof=None,
    npfact=2,
    oversample=8,
    n=1,
    search_fdot=True,
    t0=None,
    t1=None,
    silent=False,
):
    """'Quite fast folding' algorithm.

    Parameters
    ----------
    times : array of floats
        Arrival times of photons
    f0 : float
        Minimum frequency to search
    f1 : float
        Maximum frequency to search

    Other parameters
    ----------------
    nbin : int
        Number of bins to divide the profile into
    nprof : int, default None
        number of slices of the dataset to use. If None, we use 8 times nbin.
        Motivation in the comments.
    npfact : int, default 2
        maximum "sliding" of the dataset, in phase.
    oversample : int, default 8
        Oversampling wrt the standard FFT delta f = 1/T
    search_fdot : bool, default False
        Switch fdot search on or off
    t0 : float, default min(times)
        starting time
    t1 : float, default max(times)
        stop time
    """
    if nprof is None:
        # total_delta_phi = 2 == dnu * T
        # In a single sub interval
        # delta_phi = dnu * t
        # with t = T / nprof
        # so dnu T / nprof < 1 / nbin, and
        # nprof > total_delta_phi * nbin to get all the signal inside one bin
        # in a given sub-integration
        nprof = 4 * 2 * nbin * npfact

    times = copy.deepcopy(times)

    if t0 is None:
        t0 = times.min()
    if t1 is None:
        t1 = times.max()
    meantime = (t1 + t0) / 2
    times -= meantime

    maxerr = check_phase_error_after_casting_to_double(np.max(times), f1, fdot)
    if maxerr > 1 / nbin / 10:
        warnings.warn(
            f"Maximum error on the phase expected when casting to " f"double: {maxerr}"
        )
        warnings.warn(
            "Casting to double produces non-negligible phase errors. "
            "Please use shorter light curves.",
            AstropyUserWarning,
        )

    times = times.astype(np.double)

    length = t1 - t0

    frequency = (f0 + f1) / 2

    # Step: npfact * 1 / T

    step = 4 * npfact / length

    niter = int(np.rint((f1 - f0) / step)) + 2

    allvalues = list(range(-(niter // 2), niter // 2))
    if allvalues == []:
        allvalues = [0]

    all_fgrid = []
    all_fdotgrid = []
    all_stats = []

    local_show_progress = show_progress
    if silent:

        def local_show_progress(x):
            return x

    for ii, i in enumerate(local_show_progress(allvalues)):
        offset = step * i
        fdot_offset = 0

        mean_f = np.double(frequency + offset + 0.12 * step)
        mean_fdot = np.double(fdot + fdot_offset)
        mean_fddot = np.double(fddot)
        fgrid, fdotgrid, stats = search_with_qffa_step(
            times,
            mean_f,
            mean_fdot=mean_fdot,
            mean_fddot=mean_fddot,
            nbin=nbin,
            nprof=nprof,
            npfact=npfact,
            oversample=oversample,
            n=n,
            search_fdot=search_fdot,
        )

        if all_fgrid is None:
            all_fgrid = fgrid
            all_fdotgrid = fdotgrid
            all_stats = stats
        else:
            all_fgrid.append(fgrid)
            all_fdotgrid.append(fdotgrid)
            all_stats.append(stats)
    all_fgrid = np.vstack(all_fgrid)
    all_fdotgrid = np.vstack(all_fdotgrid)
    all_stats = np.vstack(all_stats)

    step = np.median(np.diff(all_fgrid[:, 0]))
    fdotstep = np.median(np.diff(all_fdotgrid[0]))
    if search_fdot:
        return (
            all_fgrid.T,
            all_fdotgrid.T,
            all_stats.T,
            step,
            fdotstep,
            length,
        )
    else:
        return all_fgrid.T[0], all_stats.T[0], step, length


def search_with_ffa(times, f0, f1, nbin=16, n=1, t0=None, t1=None):
    """'Quite fast folding' algorithm.

    Parameters
    ----------
    times : array of floats
        Arrival times of photons
    f0 : float
        Minimum frequency to search
    f1 : float
        Maximum frequency to search

    Other parameters
    ----------------
    nbin : int
        Number of bins to divide the profile into
    nprof : int, default None
        number of slices of the dataset to use. If None, we use 8 times nbin.
        Motivation in the comments.
    npfact : int, default 2
        maximum "sliding" of the dataset, in phase.
    oversample : int, default 8
        Oversampling wrt the standard FFT delta f = 1/T
    search_fdot : bool, default False
        Switch fdot search on or off
    t0 : float, default min(times)
        starting time
    t1 : float, default max(times)
        stop time
    """
    if t0 is None:
        t0 = times[0]
    if t1 is None:
        t1 = times[-1]

    length = (t1 - t0).astype(np.double)

    p0 = 1 / f1
    p1 = 1 / f0
    dt = p0 / nbin
    counts = histogram(
        (times - t0).astype(np.double),
        range=[0, length],
        bins=int(np.rint(length / dt)),
    )
    bin_periods, stats = ffa_search(counts, dt, p0, p1)
    return 1 / bin_periods, stats, None, length


def folding_search(
    events,
    fmin,
    fmax,
    step=None,
    func=epoch_folding_search,
    oversample=2,
    fdotmin=0,
    fdotmax=0,
    fdotstep=None,
    expocorr=False,
    **kwargs,
):
    times = (events.time - events.gti[0, 0]).astype(np.float64)
    weights = 1
    if hasattr(events, "counts"):
        weights = events.counts

    length = times[-1]

    if step is None:
        step = 1 / oversample / length
    if fdotstep is None:
        fdotstep = 1 / oversample / length**2
    gti = None
    if expocorr:
        gti = (events.gti - events.gti[0, 0]).astype(np.float64)

    # epsilon is needed if fmin == fmax
    epsilon = 1e-8 * step
    trial_freqs = np.arange(fmin, fmax + epsilon, step)
    fdotepsilon = 1e-2 * fdotstep
    trial_fdots = np.arange(fdotmin, fdotmax + fdotepsilon, fdotstep)
    if len(trial_fdots) > 1:
        log.info(
            "Searching {} frequencies and {} fdots".format(
                len(trial_freqs), len(trial_fdots)
            )
        )
    else:
        log.info("Searching {} frequencies".format(len(trial_freqs)))

    results = func(
        times,
        trial_freqs,
        fdots=trial_fdots,
        expocorr=expocorr,
        gti=gti,
        weights=weights,
        **kwargs,
    )
    if len(results) == 2:
        frequencies, stats = results
        return frequencies, stats, step, length
    elif len(results) == 3:
        frequencies, fdots, stats = results
        return frequencies, fdots, stats, step, fdotstep, length


def dyn_folding_search(
    events,
    fmin,
    fmax,
    step=None,
    func=epoch_folding_search,
    oversample=2,
    time_step=128,
    **kwargs,
):
    import matplotlib.pyplot as plt

    if step is None:
        step = 1 / oversample / time_step

    gti = np.copy(events.gti)
    length = np.diff(gti, axis=1)

    if not np.any(length > time_step):
        gti = np.array([[gti[0, 0], gti[-1, 1]]])

    start, stop = time_intervals_from_gtis(gti, time_step)

    stats = []

    for st, sp in zip(start, stop):
        times_filt = events.time[(events.time >= st) & (events.time < sp)]

        trial_freqs = np.arange(fmin, fmax, step)
        try:
            results = func(times_filt, trial_freqs, **kwargs)
            frequencies, stat = results
            stats.append(stat)
        except Exception:
            stats.append(np.zeros_like(trial_freqs))
    times = (start + stop) / 2
    fig = plt.figure("Dynamical search")
    plt.pcolormesh(
        frequencies.astype(float),
        times.astype(float),
        np.array(stats).astype(float),
        shading="nearest",
    )
    plt.xlabel("Frequency")
    plt.ylabel("Time")
    plt.savefig("Dyn.png")
    plt.close(fig)
    return times, frequencies, np.array(stats)


def print_qffa_results(best_cand_table):
    newtable = copy.deepcopy(best_cand_table)
    good = ~np.isnan(newtable["pulse_amp"])
    if len(newtable[good]) == 0:
        print("No pulsations found. Best candidate and upper limit:")
        good = 0
        newtable["Pulsed amplitude (%)"] = [
            f"<{a:g} (90%)" for a in newtable["pulse_amp_ul_0.9"]
        ]
    else:
        print("Best candidate(s):")
        newtable["Pulsed amplitude (%)"] = [
            f"{a:g} ± {e:g}"
            for (a, e) in zip(newtable["pulse_amp"], newtable["pulse_amp_err"])
        ]

    print(newtable["mjd", "f", "fdot", "fddot", "power", "Pulsed amplitude (%)"][good])
    return


def get_xy_boundaries_from_level(x, y, image, level, x0, y0):
    """Calculate boundaries of peaks in image.

    Parameters
    ----------
    x, y : array-like
        The coordinates of the image (anything that works with pcolormesh)
    image : 2-D array
        The image containing peaks
    level : float
        The level at which boundaries will be traced
    x0, y0 : float
        The local maximum around which the boundary has to be drawn

    Examples
    --------
    >>> x = np.linspace(-10, 10, 1000)
    >>> y = np.linspace(-10, 10, 1000)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = Z = np.sinc(np.sqrt(X**2 + Y**2))**2 + np.sinc(np.sqrt((X - 5)**2 + Y**2))**2
    >>> vals = get_xy_boundaries_from_level(X, Y, Z, 0.5, 0, 0)
    >>> assert np.allclose(np.abs(vals), 0.44, atol=0.1)
    """
    fig = plt.figure(np.random.random())
    cs = fig.gca().contour(x, y, image, [level])

    cont, seg, idx, xm, ym, d2 = find_nearest_contour(cs, x0, y0, pixel=False)

    min_x = cs.allsegs[cont][seg][:, 0].min()
    max_x = cs.allsegs[cont][seg][:, 0].max()
    min_y = cs.allsegs[cont][seg][:, 1].min()
    max_y = cs.allsegs[cont][seg][:, 1].max()
    plt.close(fig)

    return min_x, max_x, min_y, max_y


def get_boundaries_from_level(x, y, level, x0):
    """Calculate boundaries of peak in x-y plot

    Parameters
    ----------
    x, y : array-like
        The x and y values
    level : float
        The level at which boundaries will be traced
    x0 : float
        The local maximum around which the boundary has to be drawn

    Examples
    --------
    >>> x = np.linspace(-10, 10, 1000)
    >>> y = np.sinc(x)**2 + np.sinc((x - 5))**2
    >>> vals = get_boundaries_from_level(x, y, 0.5, 0)
    >>> assert np.allclose(np.abs(vals), 0.44, atol=0.1)
    """
    max_idx = np.argmin(np.abs(x - x0))
    idx = max_idx
    min_x = max_x = x0
    # lower limit
    while idx > 0 and y[idx] > level:
        min_x = x[idx]
        idx -= 1

    idx = max_idx
    # upper limit
    while idx < y.size and y[idx] > level:
        max_x = x[idx]
        idx += 1

    return min_x, max_x


def analyze_qffa_results(fname):
    """Search best candidates in a quasi-fast-folding search.

    This function searches the (typically) 2-d search plane from
    a QFFA search and finds the best five candidates.
    For the best candidate, it calculates

    Parameters
    ----------
    fname : str
        File containing the folding search results
    """
    ef = load_folding(fname)

    if not hasattr(ef, "M") or ef.M is None:
        ef.M = 1

    ntrial = ef.stat.size
    if hasattr(ef, "oversample") and ef.oversample is not None:
        ntrial /= ef.oversample
        ntrial = int(ntrial)
    if ef.kind == "Z2n":
        ndof = ef.N - 1
        detlev = z2_n_detection_level(
            epsilon=0.001,
            n=int(ef.N),
            ntrial=ntrial,
            n_summed_spectra=int(ef.M),
        )
        nbin = max(16, ef.N * 8, ef.nbin if ef.nbin is not None else 1)
        label = "$" + "Z^2_{" + f"{ef.N}" + "}$"
    else:
        ndof = ef.nbin
        detlev = fold_detection_level(nbin=int(ef.nbin), epsilon=0.001, ntrial=ntrial)
        nbin = max(16, ef.nbin)
        label = rf"$\chi^2_{ndof}$ Stat"
    n_cands = 5
    best_cands = find_peaks_in_image(ef.stat, n=n_cands)

    fddot = 0
    if hasattr(ef, "fddots") and ef.fddots is not None:
        fddot = ef.fddots

    best_cand_table = Table(
        names=[
            "fname",
            "mjd",
            "power",
            "f",
            "f_err_n",
            "f_err_p",
            "fdot",
            "fdot_err_n",
            "fdot_err_p",
            "fddot",
            "power_cl_0.9",
            "pulse_amp",
            "pulse_amp_err",
            "pulse_amp_cl_0.1",
            "pulse_amp_cl_0.9",
            "pulse_amp_ul_0.9",
            "f_idx",
            "fdot_idx",
            "fddot_idx",
        ],
        dtype=[
            str,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            int,
            int,
            int,
        ],
    )
    best_cand_table["power"].info.format = ".2f"
    best_cand_table["power_cl_0.9"].info.format = ".2f"
    best_cand_table["fdot"].info.format = ".2e"
    best_cand_table["fddot"].info.format = "g"
    best_cand_table["pulse_amp_cl_0.1"].info.format = ".2f"
    best_cand_table["pulse_amp_cl_0.9"].info.format = ".2f"
    best_cand_table["pulse_amp"].info.format = ".2f"
    best_cand_table["pulse_amp_err"].info.format = ".2f"
    best_cand_table["pulse_amp_ul_0.9"].info.format = ".2f"

    for i, idx in enumerate(best_cands):
        f_idx = fdot_idx = fddot_idx = 0
        if len(ef.stat.shape) > 1 and ef.stat.shape[0] > 1:
            f_idx, fdot_idx = idx
            allfreqs = ef.freq[f_idx, :]
            allfdots = ef.freq[:, fdot_idx]
            allstats_f = ef.stat[f_idx, :]
            allstats_fdot = ef.stat[:, fdot_idx]
            f, fdot = ef.freq[f_idx, fdot_idx], ef.fdots[f_idx, fdot_idx]
            max_stat = ef.stat[f_idx, fdot_idx]
            sig_e1_m, sig_e1 = power_confidence_limits(max_stat, c=0.68, n=ef.N)
            fmin, fmax, fdotmin, fdotmax = get_xy_boundaries_from_level(
                ef.freq, ef.fdots, ef.stat, sig_e1_m, f, fdot
            )
        elif len(ef.stat.shape) == 1:
            f_idx = idx
            allfreqs = ef.freq
            allstats_f = ef.stat
            f = ef.freq[f_idx]
            max_stat = ef.stat[f_idx]
            sig_e1_m, sig_e1 = power_confidence_limits(max_stat, c=0.68, n=ef.N)
            fmin, fmax = get_boundaries_from_level(ef.freq, ef.stat, sig_e1_m, f)
            fdot = fdotmin = fdotmax = 0
            allfdots = None
            allstats_fdot = None
        else:
            raise ValueError("Did not understand stats shape.")

        if ef.ncounts is None:
            continue

        sig_0, sig_1 = power_confidence_limits(max_stat, c=0.90, n=ef.N)
        amp = amp_err = amp_ul = amp_1 = amp_0 = np.nan
        if max_stat < detlev:
            amp_ul = a_from_ssig(sig_1, ef.ncounts) * 100
        else:
            amp = a_from_ssig(max_stat, ef.ncounts) * 100
            amp_err = a_from_ssig(sig_e1, ef.ncounts) * 100 - amp
            amp_0 = a_from_ssig(sig_0, ef.ncounts) * 100
            amp_1 = a_from_ssig(sig_1, ef.ncounts) * 100

        best_cand_table.add_row(
            [
                ef.filename,
                ef.pepoch,
                max_stat,
                f,
                fmin - f,
                fmax - f,
                fdot,
                fdotmin - fdot,
                fdotmax - fdot,
                fddot,
                sig_0,
                amp,
                amp_err,
                amp_0,
                amp_1,
                amp_ul,
                f_idx,
                fdot_idx,
                fddot_idx,
            ]
        )
        if max_stat < detlev:
            # Only add one candidate
            continue

        Table({"freq": allfreqs, "stat": allstats_f}).write(
            f'{fname.replace(HEN_FILE_EXTENSION, "")}'
            f"_cand_{n_cands - i - 1}_fdot{fdot}.csv",
            overwrite=True,
            format="ascii",
        )
        if allfdots is None:
            continue

        Table({"fdot": allfdots, "stat": allstats_fdot}).write(
            f'{fname.replace(HEN_FILE_EXTENSION, "")}'
            f"_cand_{n_cands - i - 1}_f{f}.dat",
            overwrite=True,
            format="ascii",
        )

    print_qffa_results(best_cand_table)
    best_cand_table.meta.update(
        dict(nbin=nbin, ndof=ndof, label=label, filename=None, detlev=detlev)
    )
    if (
        hasattr(ef, "filename")
        and ef.filename is not None
        and os.path.exists(ef.filename)
    ):
        best_cand_table.meta["filename"] = ef.filename

    best_cand_table.write(fname + "_best_cands.csv", overwrite=True)
    return ef, best_cand_table


def _common_parser(args=None):
    from .base import _add_default_args, check_negative_numbers_in_args

    description = "Search for pulsars using the epoch folding or the Z_n^2 " "algorithm"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs="+")
    parser.add_argument(
        "-f",
        "--fmin",
        type=float,
        required=True,
        help="Minimum frequency to fold",
    )
    parser.add_argument(
        "-F",
        "--fmax",
        type=float,
        required=True,
        help="Maximum frequency to fold",
    )
    parser.add_argument(
        "--emin",
        default=None,
        type=float,
        help="Minimum energy (or PI if uncalibrated) to plot",
    )
    parser.add_argument(
        "--emax",
        default=None,
        type=float,
        help="Maximum energy (or PI if uncalibrated) to plot",
    )
    parser.add_argument(
        "--mean-fdot",
        type=float,
        required=False,
        help="Mean fdot to fold " "(only useful when using --fast)",
        default=0,
    )
    parser.add_argument(
        "--mean-fddot",
        type=float,
        required=False,
        help="Mean fddot to fold " "(only useful when using --fast)",
        default=0,
    )
    parser.add_argument(
        "--fdotmin",
        type=float,
        required=False,
        help="Minimum fdot to fold",
        default=None,
    )
    parser.add_argument(
        "--fdotmax",
        type=float,
        required=False,
        help="Maximum fdot to fold",
        default=None,
    )
    parser.add_argument(
        "--dynstep",
        type=int,
        required=False,
        help="Dynamical EF step",
        default=128,
    )
    parser.add_argument(
        "--npfact",
        type=int,
        required=False,
        help="Size of search parameter space",
        default=2,
    )
    parser.add_argument(
        "--n-transient-intervals",
        type=int,
        required=False,
        help="Number of transient intervals to investigate",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--nbin",
        default=128,
        type=int,
        help="Number of phase bins of the profile",
    )
    parser.add_argument(
        "--segment-size",
        default=1e32,
        type=float,
        help="Size of the event list segment to use (default "
        "None, implying the whole observation)",
    )
    parser.add_argument(
        "--step",
        default=None,
        type=float,
        help="Step size of the frequency axis. Defaults to "
        "1/oversample/observ.length. ",
    )
    parser.add_argument(
        "--oversample",
        default=None,
        type=float,
        help="Oversampling factor - frequency resolution "
        "improvement w.r.t. the standard FFT's "
        "1/observ.length.",
    )
    parser.add_argument(
        "--fast",
        help="Use a faster folding algorithm. "
        "It automatically searches for the first spin "
        "derivative using an optimized step."
        "This option ignores expocorr, fdotmin/max, "
        "segment-size, and step",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--ffa",
        help="Use *the* Fast Folding Algorithm by Staelin+69. "
        "No accelerated search allowed at the moment. "
        "Only recommended to search for slow pulsars.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--transient",
        help="Look for transient emission (produces an animated"
        " GIF with the dynamic Z search)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--expocorr",
        help="Correct for the exposure of the profile bins. "
        "This method is *much* slower, but it is useful "
        "for very slow pulsars, where data gaps due to "
        "occultation or SAA passages can significantly "
        "alter the exposure of different profile bins.",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--find-candidates",
        help="Find pulsation candidates using thresholding",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--conflevel",
        default=99,
        type=float,
        help="percent confidence level for thresholding " "[0-100).",
    )

    parser.add_argument(
        "--fit-candidates",
        help="Fit the candidate peaks in the periodogram",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--curve",
        default="sinc",
        type=str,
        help="Kind of curve to use (sinc or Gaussian)",
    )
    parser.add_argument(
        "--fit-frequency",
        type=float,
        help="Force the candidate frequency to FIT_FREQUENCY",
    )

    # Only relevant to z search
    parser.add_argument(
        "-N",
        default=2,
        type=int,
        help="The number of harmonics to use in the search "
        "(the 'N' in Z^2_N; only relevant to Z search!)",
    )

    args = check_negative_numbers_in_args(args)
    _add_default_args(parser, ["deorbit", "loglevel", "debug"])

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)
    return args


def _common_main(args, func):
    args = _common_parser(args)
    files = args.files
    if args.fit_candidates and args.fit_frequency is None:
        args.find_candidates = True
    elif args.fit_candidates and args.fit_frequency is not None:
        args.find_candidates = False

    if func != z_n_search and args.fast:
        raise ValueError("The fast option is only available for z searches")

    outfiles = []
    for i_f, fname in enumerate(files):
        log.info(f"Treating {fname}")
        mjdref = 0
        kwargs = {}
        baseline = args.nbin
        kind = "EF"
        kind_label = kind
        n = 1
        if func == z_n_search:
            n = args.N
            kwargs = {"nharm": args.N}
            baseline = args.N
            kind = "Z2n"
            kind_label = f"Z2{n}"
        ftype, events = get_file_type(fname)

        out_fname = hen_root(fname) + "_{}".format(kind_label)
        if args.emin is not None or args.emax is not None:
            emin = assign_value_if_none(args.emin, HENDRICS_STAR_VALUE)
            emax = assign_value_if_none(args.emax, HENDRICS_STAR_VALUE)
            out_fname += f"_{emin:g}-{emax:g}keV"
        if args.fmin is not None or args.fmax is not None:
            fmin = assign_value_if_none(args.fmin, HENDRICS_STAR_VALUE)
            fmax = assign_value_if_none(args.fmax, HENDRICS_STAR_VALUE)
            out_fname += f"_{fmin:g}-{fmax:g}Hz"
        if args.fast:
            out_fname += "_fast"
        elif args.ffa:
            out_fname += "_ffa"
        if args.mean_fdot is not None and not np.isclose(args.mean_fdot * 1e10, 0):
            out_fname += f"_fd{args.mean_fdot * 1e10:g}e-10s-2"

        if ftype == "events":
            if hasattr(events, "mjdref"):
                mjdref = events.mjdref
            if args.emin is not None or args.emax is not None:
                events, elabel = filter_energy(events, args.emin, args.emax)

            if args.deorbit_par is not None:
                events = deorbit_events(events, args.deorbit_par)

        if args.fast:
            oversample = assign_value_if_none(args.oversample, 4 * n)
        else:
            oversample = assign_value_if_none(args.oversample, 2)

        if args.transient and ftype == "lc":
            log.error("Transient search not yet available for light curves")
        if args.transient and ftype == "events":
            results = transient_search(
                events.time,
                args.fmin,
                args.fmax,
                fdot=0,
                nbin=args.nbin,
                n=n,
                nprof=args.n_transient_intervals,
                oversample=oversample,
            )
            plot_transient_search(results, out_fname + "_transient.gif")
            continue

        if not args.fast and not args.ffa:
            fdotmin = args.fdotmin if args.fdotmin is not None else 0
            fdotmax = args.fdotmax if args.fdotmax is not None else 0
            results = folding_search(
                events,
                args.fmin,
                args.fmax,
                step=args.step,
                func=func,
                oversample=oversample,
                nbin=args.nbin,
                expocorr=args.expocorr,
                fdotmin=fdotmin,
                fdotmax=fdotmax,
                segment_size=args.segment_size,
                **kwargs,
            )
            ref_time = events.gti[0, 0]
        elif args.fast:
            fdotmin = args.fdotmin if args.fdotmin is not None else 0
            fdotmax = args.fdotmax if args.fdotmax is not None else 0
            search_fdot = True
            if args.fdotmax is not None and fdotmax <= fdotmin:
                search_fdot = False
            nbin = args.nbin
            if nbin / n < 8:
                nbin = n * 8
                warnings.warn(
                    f"The number of bins is too small for Z search."
                    f"Increasing to {nbin}"
                )
            results = search_with_qffa(
                events.time,
                args.fmin,
                args.fmax,
                fdot=args.mean_fdot,
                fddot=args.mean_fddot,
                nbin=nbin,
                n=n,
                nprof=None,
                npfact=args.npfact,
                oversample=oversample,
                search_fdot=search_fdot,
            )

            ref_time = (events.time[-1] + events.time[0]) / 2
        elif args.ffa:
            warnings.warn(
                "The Fast Folding Algorithm functionality is experimental. Use"
                " with care, and feel free to report any issues."
            )
            results = search_with_ffa(
                events.time, args.fmin, args.fmax, nbin=args.nbin, n=n
            )
            ref_time = events.time[0]

        length = events.time.max() - events.time.min()
        segment_size = np.min([length, args.segment_size])
        M = length // segment_size

        fdots = 0
        if len(results) == 4:
            frequencies, stats, step, length = results
        elif len(results) == 6:
            frequencies, fdots, stats, step, fdotsteps, length = results

        if length > args.dynstep and not (args.fast or args.ffa):
            _ = dyn_folding_search(
                events,
                args.fmin,
                args.fmax,
                step=step,
                func=func,
                oversample=oversample,
                time_step=args.dynstep,
                **kwargs,
            )

        efperiodogram = EFPeriodogram(
            frequencies,
            stats,
            kind,
            args.nbin,
            args.N,
            fdots=fdots,
            M=M,
            segment_size=segment_size,
            filename=fname,
            parfile=args.deorbit_par,
            emin=args.emin,
            emax=args.emax,
            mjdref=mjdref,
            pepoch=mjdref + ref_time / 86400,
            oversample=args.oversample,
        )
        efperiodogram.upperlim = pf_upper_limit(
            np.max(stats), events.time.size, n=args.N
        )
        efperiodogram.ncounts = events.time.size
        best_peaks = None
        if args.find_candidates:
            best_peaks, best_stat = efperiodogram.find_peaks(conflevel=args.conflevel)
        elif args.fit_frequency is not None:
            best_peaks = np.array([args.fit_frequency])
            efperiodogram.peaks = best_peaks
            efperiodogram.peak_stat = [0]

        best_models = []
        detected = best_peaks is not None and len(best_peaks) > 0

        if args.fit_candidates and not detected:
            warnings.warn("No peaks detected")
        elif args.fit_candidates and not (args.fast or args.ffa):
            search_width = 5 * oversample * step
            for f in best_peaks:
                good = np.abs(frequencies - f) < search_width
                if args.curve.lower() == "sinc":
                    best_fun = fit(
                        frequencies[good],
                        stats[good],
                        f,
                        obs_length=length,
                        baseline=baseline,
                    )
                elif args.curve.lower() == "gaussian":
                    best_fun = fit(frequencies[good], stats[good], f, baseline=baseline)
                else:
                    raise ValueError("`--curve` arg must be sinc or gaussian")

                best_models.append(best_fun)

        efperiodogram.best_fits = best_models
        efperiodogram.oversample = oversample

        save_folding(efperiodogram, out_fname + HEN_FILE_EXTENSION)
        if args.fast:
            analyze_qffa_results(out_fname + HEN_FILE_EXTENSION)

        outfiles.append(out_fname + HEN_FILE_EXTENSION)
    return outfiles


def main_efsearch(args=None):
    """Main function called by the `HENefsearch` command line script."""

    with log.log_to_file("HENefsearch.log"):
        return _common_main(args, epoch_folding_search)


def main_zsearch(args=None):
    """Main function called by the `HENzsearch` command line script."""

    with log.log_to_file("HENzsearch.log"):
        return _common_main(args, z_n_search)


def z2_vs_pf(event_list, deadtime=0.0, ntrials=100, outfile=None, N=2):
    length = event_list.gti[-1, 1] - event_list.gti[0, 0]
    df = 1 / length

    result_table = Table(names=["pf", "z2"], dtype=[float, float])
    for i in show_progress(range(ntrials)):
        pf = np.random.uniform(0, 1)
        new_event_list = scramble(
            event_list,
            deadtime=deadtime,
            smooth_kind="pulsed",
            pulsed_fraction=pf,
        )
        frequencies, stats, _, _ = search_with_qffa(
            new_event_list.time,
            1 - df * 2,
            1 + df * 2,
            fdot=0,
            nbin=32,
            oversample=16,
            search_fdot=False,
            silent=True,
            n=N,
        )
        result_table.add_row([pf, np.max(stats)])
    if outfile is None:
        outfile = "z2_vs_pf.csv"
    result_table.write(outfile, overwrite=True)
    return result_table


def main_z2vspf(args=None):
    from .base import _add_default_args, check_negative_numbers_in_args

    description = (
        "Get Z2 vs pulsed fraction for a given observation. Takes"
        " the original event list, scrambles the event arrival time,"
        " adds a pulsation with random pulsed fraction, and takes"
        " the maximum value of Z2 in a small interval around the"
        " pulsation. Does this ntrial times, and plots."
    )
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("fname", help="Input file name")
    parser.add_argument(
        "--ntrial",
        default=100,
        type=int,
        help="Number of trial values for the pulsed fraction",
    )
    parser.add_argument(
        "--outfile", default=None, type=str, help="Output table file name"
    )
    parser.add_argument(
        "--show-z-values",
        nargs="+",
        default=None,
        type=float,
        help="Show these Z values in the plot",
    )
    parser.add_argument(
        "--emin",
        default=None,
        type=float,
        help="Minimum energy (or PI if uncalibrated) to plot",
    )
    parser.add_argument(
        "--emax",
        default=None,
        type=float,
        help="Maximum energy (or PI if uncalibrated) to plot",
    )
    parser.add_argument("-N", default=2, type=int, help="The N in Z^2_N")

    args = check_negative_numbers_in_args(args)
    _add_default_args(parser, ["loglevel", "debug"])

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)

    outfile = args.outfile
    if outfile is None:
        outfile = hen_root(args.fname) + "_z2vspf.csv"

    events = load_events(args.fname)
    if args.emin is not None or args.emax is not None:
        events, elabel = filter_energy(events, args.emin, args.emax)

    result_table = z2_vs_pf(
        events, deadtime=0.0, ntrials=args.ntrial, outfile=outfile, N=args.N
    )
    if HAS_MPL:
        fig = plt.figure("Results", figsize=(10, 6))
        plt.scatter(result_table["pf"] * 100, result_table["z2"])
        plt.semilogy()
        plt.grid(True)
        plt.xlabel(r"Pulsed fraction (%)")
        plt.ylabel(r"$Z^2_{}$".format(args.N))
        # plt.show()
        if args.show_z_values is not None:
            for z in args.show_z_values:
                plt.axhline(z, alpha=0.5, color="r", ls="--")
        plt.savefig(outfile.replace(".csv", ".png"))
        plt.close(fig)


def main_accelsearch(args=None):
    from stingray.pulse.accelsearch import accelsearch

    from .base import _add_default_args, check_negative_numbers_in_args

    warnings.warn(
        "The accelsearch functionality is experimental. Use with care, "
        " and feel free to report any issues."
    )
    description = "Run the accelerated search on pulsar data."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("fname", help="Input file name")
    parser.add_argument("--outfile", default=None, type=str, help="Output file name")
    parser.add_argument(
        "--emin",
        default=None,
        type=float,
        help="Minimum energy (or PI if uncalibrated) to plot",
    )
    parser.add_argument(
        "--emax",
        default=None,
        type=float,
        help="Maximum energy (or PI if uncalibrated) to plot",
    )
    parser.add_argument(
        "--fmin",
        default=0.1,
        type=float,
        help="Minimum frequency to search, in Hz",
    )
    parser.add_argument(
        "--fmax",
        default=1000,
        type=float,
        help="Maximum frequency to search, in Hz",
    )
    parser.add_argument(
        "--nproc", default=1, type=int, help="Number of processors to use"
    )
    parser.add_argument(
        "--zmax",
        default=100,
        type=int,
        help="Maximum acceleration (in spectral bins)",
    )
    parser.add_argument(
        "--delta-z",
        default=1,
        type=float,
        help="Fdot step for search (1 is the default resolution)",
    )
    parser.add_argument(
        "--interbin",
        default=False,
        action="store_true",
        help="Use interbinning",
    )
    parser.add_argument(
        "--pad-to-double",
        default=False,
        action="store_true",
        help="Pad to the double of bins " "(sort-of interbinning)",
    )
    parser.add_argument(
        "--detrend",
        default=None,
        type=float,
        help="Detrending timescale",
    )
    parser.add_argument(
        "--deorbit-par",
        default=None,
        type=str,
        help="Parameter file in TEMPO2/PINT format",
    )
    parser.add_argument(
        "--red-noise-filter",
        default=False,
        action="store_true",
        help="Correct FFT for red noise (use with caution)",
    )

    args = check_negative_numbers_in_args(args)
    _add_default_args(parser, ["loglevel", "debug"])

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)

    outfile = args.outfile
    if outfile is None:
        label = "_accelsearch"
        if args.emin is not None or args.emax is not None:
            emin = assign_value_if_none(args.emin, HENDRICS_STAR_VALUE)
            emax = assign_value_if_none(args.emax, HENDRICS_STAR_VALUE)
            label += f"_{emin:g}-{emax:g}keV"

        if args.interbin:
            label += "_interbin"
        elif args.pad_to_double:
            label += "_pad"

        if args.red_noise_filter:
            label += "_rednoise"
        if args.detrend:
            label += f"_detrend{args.detrend}"

        outfile = hen_root(args.fname) + label + ".csv"

    emin = args.emin
    emax = args.emax
    debug = args.debug
    interbin = args.interbin
    zmax = args.zmax
    fmax = args.fmax
    fmin = args.fmin
    delta_z = args.delta_z
    nproc = args.nproc

    log.info(f"Opening file {args.fname}")
    events = load_events(args.fname)

    if args.deorbit_par is not None:
        events = deorbit_events(events, args.deorbit_par)

    nyq = fmax * 5
    dt = 0.5 / nyq
    log.info(f"Searching using dt={dt}")

    if emin is not None or emax is not None:
        events, elabel = filter_energy(events, emin, emax)

    tstart = events.gti[0, 0]
    GTI = events.gti
    max_length = GTI.max() - tstart
    event_times = events.time

    t0 = GTI[0, 0]
    Nbins = int(np.rint(max_length / dt))
    if Nbins > 10**8:
        log.info(
            f"The number of bins is more than 100 millions: {Nbins}. " "Using memmap."
        )

    dt = adjust_dt_for_power_of_two(dt, max_length)

    if args.pad_to_double:
        times = memmapped_arange(-0.5 * max_length, 1.5 * max_length, dt)
        counts = histogram(
            (event_times - t0).astype(np.double),
            bins=times.size,
            range=[
                -np.double(max_length) * 0.5,
                np.double(max_length - dt) * 1.5,
            ],
        )
    else:
        times = memmapped_arange(0, max_length, dt)
        counts = histogram(
            (event_times - t0).astype(np.double),
            bins=times.size,
            range=[0, np.double(max_length - dt)],
        )

    if args.detrend is not None:
        log.info("Detrending light curve")
        Nsmooth = args.detrend / dt / 3
        plt.figure("Bu")
        plt.plot(times, counts)
        for g in GTI - t0:
            print(g, Nsmooth)
            good = (times > g[0]) & (times <= g[1])
            if (g[1] - g[0]) < args.detrend:
                counts[good] = 0
            else:
                counts[good] -= gaussian_filter(counts[good], Nsmooth, mode="reflect")
        counts += 2
        plt.plot(times, counts)
        plt.show()

    log.info(f"Times and counts have {times.size} bins")
    # Note: det_p_value was calculated as
    # pds_probability(pds_detection_level(0.015) * 0.64) => 0.068
    # where 0.64 indicates the 36% detection level drop at the bin edges.
    # Interbin multiplies the number of candidates, hence use the standard
    # detection level
    det_p_value = 0.068
    if interbin:
        det_p_value = 0.015
    elif args.pad_to_double:
        # Half of the bins are zeros.
        det_p_value = 0.068 * 2

    fft_rescale = None
    if args.red_noise_filter:

        def fft_rescale(fourier_trans):
            pds = (fourier_trans * fourier_trans.conj()).real
            smooth = gaussian_filter(pds, 31)
            rescale = 2 / smooth
            return fourier_trans * rescale**0.5

    results = accelsearch(
        times,
        counts,
        delta_z=delta_z,
        fmin=fmin,
        fmax=fmax,
        gti=GTI - t0,
        zmax=zmax,
        ref_time=t0,
        debug=debug,
        interbin=interbin,
        nproc=nproc,
        det_p_value=det_p_value,
        fft_rescale=fft_rescale,
        candidate_file=outfile.replace(".csv", ""),
    )

    if len(results) > 0:
        results["emin"] = emin if emin else -1.0
        results["emax"] = emax if emax else -1.0
        results["fmin"] = fmin
        results["fmax"] = fmax
        results["zmax"] = zmax
        if hasattr(events, "mission"):
            results["mission"] = events.mission.replace(",", "+")
        results["instr"] = events.instr.replace(",", "+")
        results["mjdref"] = np.double(events.mjdref)
        results["pepoch"] = events.mjdref + results["time"] / 86400.0

        results.sort("power")

        print("Best candidates:")
        results["time", "frequency", "fdot", "power", "pepoch"][-10:][::-1].pprint()
        print(f"See all {len(results)} candidates in {outfile}")
    else:
        print("No candidates found")

    log.info("Writing results to file")
    results.write(outfile, overwrite=True)

    return outfile
