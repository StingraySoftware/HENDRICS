# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Search for pulsars."""

import warnings
import os
import argparse
import copy

import numpy as np
from astropy import log
from astropy.table import Table
from astropy.logger import AstropyUserWarning
from .io import get_file_type
from stingray.pulse.search import (
    epoch_folding_search,
    z_n_search,
    search_best_peaks,
)
from stingray.gti import time_intervals_from_gtis
from stingray.utils import assign_value_if_none
from stingray.pulse.modeling import fit_sinc, fit_gaussian
from .io import load_events, EFPeriodogram, save_folding, HEN_FILE_EXTENSION
from .base import hen_root, show_progress, adjust_dt_for_power_of_two
from .base import deorbit_events, njit, prange
from .base import histogram2d, histogram, memmapped_arange
from .base import z2_n_detection_level
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
    phase = times * f + 0.5 * times ** 2 * fdot
    phase_dbl = times_dbl * np.double(f) + 0.5 * times_dbl ** 2 * np.double(
        fdot
    )
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
        "Recommended frequency steps: {}".format(
            int(np.diff(freq_range)[0] // df + 1)
        )
    )
    while count < NMAX:
        # In any case, only the first loop deletes the file
        if count > 0:
            reset = False
        block_of_data = []
        freq = np.random.uniform(freq_range[0], freq_range[1])
        fdot = np.random.uniform(fdot_range[0], fdot_range[1])

        dX = 1 / (TWOPI * freq)

        nX = np.int(np.diff(asini_range) // dX) + 1
        Xs = np.random.uniform(asini_range[0], asini_range[1], nX)

        for X in Xs:
            dOmega = 1 / (TWOPI * freq * X * length) * D_OMEGA_FACTOR
            nOmega = np.int(np.diff(omega_range) // dOmega) + 1
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

            dT0 = min(1 / (TWOPI ** 2 * freq) * Porb / X, Porb / 10)
            max_stats = 0
            min_stats = 1e32
            best_T0 = None
            T0s = np.random.uniform(0, Porb, int(Porb // dT0 + 1))
            for T0 in T0s:
                # one iteration
                new_values = times - X * np.sin(
                    2 * np.pi * (times - T0) / Porb
                )
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


def fit(
    frequencies, stats, center_freq, width=None, obs_length=None, baseline=0
):
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
def calculate_shifts(
    nprof: int, nbin: int, nshift: int, order: int = 1
) -> np.array:
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
        total_shift_int = np.int(total_shift)

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
    >>> np.isclose(z_n_fast(phase, norm, n=4), 50)
    True
    >>> np.isclose(z_n_fast(phase, norm, n=2), 50)
    True
    """

    total_norm = np.sum(norm)

    result = 0
    # Instead of calculating k phi each time
    kph = np.zeros_like(phase)

    for k in range(1, n + 1):
        kph += phase
        result += (
            np.sum(np.cos(kph) * norm) ** 2 + np.sum(np.sin(kph) * norm) ** 2
        )

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
    >>> np.isclose(results[0, 8], 50)
    True
    >>> np.isclose(results[1, 8], 50/2)
    True
    >>> np.isclose(results[2, 8], 50/4)
    True
    >>> np.isclose(results[3, 8], 50/8)
    True
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
        shape_0 = np.int(profiles.shape[0] / n_ave_i)
        # new_profiles = np.zeros((shape_0, profiles.shape[1]))
        for i in range(shape_0):
            new_profiles = np.sum(
                profiles[i * n_ave_i : (i + 1) * n_ave_i], axis=0
            )

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
        f"Maximum error on the phase expected when casting to double: "
        f"{maxerr}"
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
    results.stats = np.array(
        [all_results[:, i, :].T for i in range(nave.size)]
    )

    return results


def plot_transient_search(results, gif_name=None):
    import matplotlib.pyplot as plt

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
        gs = plt.GridSpec(2, 2, height_ratios=(1, 3))
        for i_f in [0, 1]:
            axf = plt.subplot(gs[0, i_f])
            axima = plt.subplot(gs[1, i_f], sharex=axf)

            axima.pcolormesh(f, t, ima / detl * 3, vmax=3, vmin=0.3)

            mean_line = np.mean(ima, axis=0) / sum_detl * 3
            maxidx = np.argmax(mean_line)
            maxline = mean_line[maxidx]
            best_f = f[maxidx]
            for il, line in enumerate(ima / detl * 3):
                axf.plot(
                    f, line, lw=0.2, ls="-", c="grey", alpha=0.5, label=f"{il}"
                )
                maxidx = np.argmax(mean_line)
                if line[maxidx] > maxline:
                    best_f = f[maxidx]
                    maxline = line[maxidx]

            axf.plot(
                f, mean_line, lw=1, c="k", zorder=10, label="mean", ls="-"
            )

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
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        all_images.append(image)

    if HAS_IMAGEIO:
        imageio.mimsave(gif_name, all_images, fps=1)
    else:
        warnings.warn(
            "imageio needed to save the transient search results "
            "into a gif image."
        )

    return all_images


@njit(nogil=True)
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
    quad_base_shift = base_shift ** 2

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
            local_stat = _z_n_fast_cached(
                splat_prof, cached_cos, cached_sin, n=n
            )
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
    phases = (
        ts * mean_f
        + 0.5 * tssq * mean_fdot
        + ONE_SIXTH * tssq * ts * mean_fddot
    )
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
    linbinshifts = np.linspace(
        -nbin * npfact, nbin * npfact, int(oversample * npfact)
    )
    if search_fdot:
        quabinshifts = np.linspace(
            -nbin * npfact, nbin * npfact, int(oversample * npfact)
        )
    else:
        quabinshifts = np.array([0])

    dphi = 1 / nbin
    delta_t = (t1 - t0) / 2
    bin_to_frequency = dphi / delta_t
    bin_to_fdot = 2 * dphi / delta_t ** 2

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
            f"Maximum error on the phase expected when casting to "
            f"double: {maxerr}"
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
        return all_fgrid.T, all_fdotgrid.T, all_stats.T, step, fdotstep, length
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
        fdotstep = 1 / oversample / length ** 2
    gti = None
    if expocorr:
        gti = events.gti

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
    plt.pcolormesh(frequencies, times, np.array(stats))
    plt.xlabel("Frequency")
    plt.ylabel("Time")
    plt.savefig("Dyn.png")
    plt.close(fig)
    return times, frequencies, np.array(stats)


def _common_parser(args=None):
    from .base import _add_default_args, check_negative_numbers_in_args

    description = (
        "Search for pulsars using the epoch folding or the Z_n^2 " "algorithm"
    )
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
        best_peaks = [args.fit_frequency]

    if func != z_n_search and args.fast:
        raise ValueError("The fast option is only available for z searches")

    outfiles = []
    for i_f, fname in enumerate(files):
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
                nprof=None,
                oversample=oversample,
            )
            plot_transient_search(results, hen_root(fname) + "_transient.gif")
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
        )

        if args.find_candidates:
            threshold = 1 - args.conflevel / 100
            best_peaks, best_stat = search_best_peaks(
                frequencies, stats, threshold
            )
            efperiodogram.peaks = best_peaks
            efperiodogram.peak_stat = best_stat
        elif args.fit_frequency is not None:
            efperiodogram.peaks = best_peaks
            efperiodogram.peak_stat = [0]

        best_models = []

        if args.fit_candidates and not (args.fast or args.ffa):
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
                    best_fun = fit(
                        frequencies[good], stats[good], f, baseline=baseline
                    )
                else:
                    raise ValueError("`--curve` arg must be sinc or gaussian")

                best_models.append(best_fun)

        efperiodogram.best_fits = best_models
        efperiodogram.oversample = oversample

        out_fname = hen_root(fname) + "_{}".format(kind_label)
        if args.emin is not None or args.emax is not None:
            emin = assign_value_if_none(args.emin, "**")
            emax = assign_value_if_none(args.emax, "**")
            out_fname += f"_{emin:g}-{emax:g}keV"
        if args.fmin is not None or args.fmax is not None:
            fmin = assign_value_if_none(args.fmin, "**")
            fmax = assign_value_if_none(args.fmax, "**")
            out_fname += f"_{fmin:g}-{fmax:g}Hz"
        if args.fast:
            out_fname += "_fast"
        elif args.ffa:
            out_fname += "_ffa"
        if args.mean_fdot is not None and not np.isclose(
            args.mean_fdot * 1e10, 0
        ):
            out_fname += f"_fd{args.mean_fdot * 1e10:g}e-10s-2"

        save_folding(efperiodogram, out_fname + HEN_FILE_EXTENSION)
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
        plt.figure("Results", figsize=(10, 6))
        plt.scatter(result_table["pf"] * 100, result_table["z2"])
        plt.semilogy()
        plt.grid(True)
        plt.xlabel(r"Pulsed fraction (%)")
        plt.ylabel(r"$Z^2_2$")
        # plt.show()
        plt.savefig(outfile.replace(".csv", ".png"))


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
    parser.add_argument(
        "--outfile", default=None, type=str, help="Output file name"
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
        "--delta-z", default=1, type=int, help="Fdot step for search"
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
            emin = assign_value_if_none(args.emin, "**")
            emax = assign_value_if_none(args.emax, "**")
            label += f"_{emin:g}-{emax:g}keV"
        if args.interbin:
            label += "_interbin"
        elif args.pad_to_double:
            label += "_pad"

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
    if Nbins > 10 ** 8:
        log.info(
            f"The number of bins is more than 100 millions: {Nbins}. "
            "Using memmap."
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
    )

    if len(results) > 0:
        results["emin"] = emin if emin else -1.0
        results["emax"] = emax if emax else -1.0
        results["fmin"] = fmin
        results["fmax"] = fmax
        results["zmax"] = zmax
        if hasattr(events, "mission"):
            results["mission"] = events.mission
        results["instr"] = events.instr
        results["mjdref"] = np.double(events.mjdref)
        results["pepoch"] = events.mjdref + results["time"] / 86400.0

        results.sort("power")

        print("Best candidates:")
        results["time", "frequency", "fdot", "power", "pepoch"][-10:][
            ::-1
        ].pprint()
        print(f"See all {len(results)} candidates in {outfile}")
    else:
        print("No candidates found")

    results.write(outfile, overwrite=True)

    return outfile
