# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Search for pulsars."""

import warnings
import os
import argparse
from functools import wraps
import copy
import numpy as np
from astropy import log
from astropy.logger import AstropyUserWarning
from stingray.pulse.search import epoch_folding_search, z_n_search, \
    search_best_peaks
from stingray.gti import time_intervals_from_gtis
from stingray.utils import assign_value_if_none
from stingray.pulse.modeling import fit_sinc, fit_gaussian
from .io import load_events, EFPeriodogram, save_folding, \
    HEN_FILE_EXTENSION
from .base import hen_root
from .fold import filter_energy


try:
    from fast_histogram import histogram2d
    HAS_FAST_HIST = True
except ImportError:
    from numpy import histogram2d as histogram2d_np

    def histogram2d(*args, **kwargs):
        return histogram2d_np(*args, **kwargs)[0]

try:
    from numba import njit, prange
except ImportError:
    def njit(**kwargs):
        """Dummy decorator in case jit cannot be imported."""
        def true_decorator(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                r = f(*args, **kwargs)
                return r
            return wrapped
        return true_decorator

    def prange(*args):
        """Dummy decorator in case jit cannot be imported."""
        return range(*args)

from .base import deorbit_events


try:
    from tqdm import tqdm as show_progress
except ImportError:
    def show_progress(a):
        return a


D_OMEGA_FACTOR = 2 * np.sqrt(3)
TWOPI = 2 * np.pi


def _save_df_to_csv(df, csv_file, reset=False):
    if not os.path.exists(csv_file) or reset:
        mode = 'w'
        header = True
    else:
        mode = 'a'
        header = False
    df.to_csv(csv_file, header=header, index=False, mode=mode)


def check_phase_error_after_casting_to_double(tref, f, fdot=0):
    """Check the maximum error expected in the phase when casting to double."""
    times = np.array(np.random.normal(tref, 0.1, 1000), dtype=np.longdouble)
    times_dbl = times.astype(np.double)
    phase = times * f + 0.5 * times * fdot ** 2
    phase_dbl = times_dbl * np.double(f) + \
        0.5 * times_dbl ** 2 * np.double(fdot)
    return np.max(np.abs(phase_dbl - phase))


def decide_binary_parameters(length, freq_range, porb_range, asini_range,
                             fdot_range=[0, 0], NMAX=10,
                             csv_file='db.csv', reset=False):
    import pandas as pd
    count = 0
    omega_range = [1 / porb_range[1], 1 / porb_range[0]]
    columns = [
        'freq',
        'fdot',
        'X',
        'Porb',
        'done',
        'max_stat',
        'min_stat',
        'best_T0']

    df = 1 / length
    print('Recommended frequency steps: {}'.format(
        int(np.diff(freq_range)[0] // df + 1)))
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
            Omegas = np.random.uniform(omega_range[0], omega_range[1],
                                       nOmega)

            for Omega in Omegas:
                block_of_data.append([freq, fdot, X, TWOPI / Omega,
                                      False, 0., 0., 0.])

        df = pd.DataFrame(block_of_data, columns=columns)
        _save_df_to_csv(df, csv_file, reset=reset)
        count += 1

    return csv_file


def folding_orbital_search(events, parameter_csv_file, chunksize=100,
                           outfile='out.csv',
                           fun=epoch_folding_search, **fun_kwargs):
    import pandas as pd

    times = (events.time - events.gti[0, 0]).astype(np.float64)
    for chunk in pd.read_csv(parameter_csv_file, chunksize=chunksize):
        try:
            chunk['done'][0]
        except Exception:
            continue
        for i in range(len(chunk)):
            if chunk['done'][i]:
                continue

            row = chunk.iloc[i]
            freq, fdot, X, Porb = \
                np.array([row['freq'], row['fdot'], row['X'], row['Porb']],
                         dtype=np.float64)

            dT0 = min(1 / (TWOPI ** 2 * freq) * Porb / X, Porb / 10)
            max_stats = 0
            min_stats = 1e32
            best_T0 = None
            T0s = np.random.uniform(0, Porb, int(Porb // dT0 + 1))
            for T0 in T0s:
                # one iteration
                new_values = \
                    times - X * np.sin(2 * np.pi * (times - T0) / Porb)
                new_values = \
                    new_values - X * np.sin(2 * np.pi *
                                            (new_values - T0) / Porb)
                fgrid, stats = \
                    fun(new_values, np.array([freq]), fdots=fdot, **fun_kwargs)
                if stats[0] > max_stats:
                    max_stats = stats[0]
                    best_T0 = T0
                if stats[0] < min_stats:
                    min_stats = stats[0]
            idx = chunk.index[i]
            chunk.iloc[idx, chunk.columns.get_loc('max_stat')] = max_stats
            chunk.iloc[idx, chunk.columns.get_loc('min_stat')] = min_stats
            chunk.iloc[idx, chunk.columns.get_loc('best_T0')] = best_T0

            chunk.iloc[idx, chunk.columns.get_loc('done')] = True
        _save_df_to_csv(chunk, outfile)


def fit(frequencies, stats, center_freq, width=None, obs_length=None,
        baseline=0):
    estimated_amp = stats[np.argmin(np.abs(frequencies - center_freq))]

    if obs_length is not None:
        s = fit_sinc(frequencies, stats - baseline, obs_length=obs_length,
                     amp=estimated_amp, mean=center_freq)
    else:
        df = frequencies[1] - frequencies[0]
        if width is None:
            width = 2 * df
        s = fit_gaussian(frequencies, stats - baseline, stddev=width,
                         amplitude=estimated_amp, mean=center_freq)

    return s


@njit()
def calculate_shifts(
        nprof: int,
        nbin: int,
        nshift: int,
        order: int = 1) -> np.array:
    shifts = np.linspace(-1., 1., nprof) ** order
    return nshift * shifts


@njit()
def shift_and_select(repeated_profiles, lshift, qshift, newprof):
    nprof = len(repeated_profiles)
    nbin = len(newprof[0])
    lshifts = calculate_shifts(nprof, nbin, lshift, 1)
    qshifts = calculate_shifts(nprof, nbin, qshift, 2)
    for k in range(nprof):
        total_shift = int(np.rint(lshifts[k] + qshifts[k])) % nbin
        newprof[k, :] = repeated_profiles[k, nbin -
                                          total_shift: 2 * nbin - total_shift]
    return newprof


@njit(fastmath=True)
def z_n_fast(phase, norm, n=2):
    '''Z^2_n statistics, a` la Buccheri+03, A&A, 128, 245, eq. 2.

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
    '''

    total_norm = np.sum(norm)

    result = 0
    # Instead of calculating k phi each time
    kph = np.zeros_like(phase)

    for k in range(1, n + 1):
        kph += phase
        result += np.sum(np.cos(kph) * norm) ** 2 + \
            np.sum(np.sin(kph) * norm) ** 2

    return 2 / total_norm * result


@njit()
def _average_and_z_sub_search(profiles, n=2):
    '''Z^2_n statistics calculated in sub-profiles.

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
    '''
    nprof = len(profiles)
    # Only use powers of two
    nprof = int(2 ** np.log2(nprof))
    profiles = profiles[:nprof]

    nbin = len(profiles[0])

    n_log_ave_max = int(np.log2(nprof))

    results = np.zeros((n_log_ave_max, nprof))
    all_nave = np.zeros((n_log_ave_max, nprof))

    twopiphases = 2 * np.pi * np.arange(0, 1, 1 / nbin)

    n_ave = 2**np.arange(n_log_ave_max)

    for ave_i in range(len(n_ave)):
        n_ave_i = n_ave[ave_i]
        shape_0 = np.int(profiles.shape[0] / n_ave_i)
        # new_profiles = np.zeros((shape_0, profiles.shape[1]))
        for i in range(shape_0):
            new_profiles = np.sum(profiles[i * n_ave_i: (i + 1) * n_ave_i], axis=0)
            if np.max(new_profiles) == 0:
                continue

            z = z_n_fast(twopiphases, norm=new_profiles, n=n)
            results[ave_i, i * n_ave_i: (i + 1) * n_ave_i] = z

    return n_ave, results


def _transient_search_step(
        times: np.double,
        mean_f: np.double,
        mean_fdot=0,
        nbin=16,
        nprof=64,
        n=1):
    """Single step of transient search."""

    # Cast to standard double, or the fast_histogram.histogram2d will fail
    # horribly.

    phases = _fast_phase_fdot(times, mean_f, mean_fdot)

    profiles = histogram2d(phases, times, range=[
        [0, 1], [times[0], times[-1]]], bins=(nbin, nprof)).T

    n_ave, results = _average_and_z_sub_search(profiles, n=n)
    return n_ave, results


class TransientResults(object):
    pass


def transient_search(times, f0, f1, fdot=0, nbin=16, nprof=None, n=1,
                     t0=None, t1=None, oversample=4):
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
        f"Maximum error on the phase expected when casting to double: {maxerr}")
    if maxerr > 1 / nbin / 10:
        warnings.warn(
            "Casting to double produces non-negligible phase errors. "
            "Please use shorter light curves.",
            AstropyUserWarning)

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
        nave, results = \
            _transient_search_step(times, mean_f, mean_fdot=mean_fdot,
                                   nbin=nbin, nprof=nprof, n=n)
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
    from hendrics.fold import z2_n_detection_level
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False
    if gif_name is None:
        gif_name = 'transients.gif'

    all_images = []
    for i, (ima, nave) in enumerate(zip(results.stats, results.nave)):
        f = results.freqs
        t = results.times
        nprof = ima.shape[0]
        oversample = results.oversample

        # To calculate ntrial, we need to take into account that
        # 1. the image has nave equal pixels
        # 2. the frequency axis is oversampled by at least nprof / nave
        ntrial = ima.size / nave / (nprof / nave) / oversample

        detl = z2_n_detection_level(0.0015, n=2,
                                    ntrial=ntrial)

        # To calculate ntrial from the summed image, we use the
        # length of the frequency axis, considering oversample by
        # nprof / nave:
        ntrial_sum = f.size / nave / (nprof / nave) / oversample

        sum_detl = z2_n_detection_level(0.0015, n=2,
                                        ntrial=ntrial_sum,
                                        n_summed_spectra=nprof/nave)
        fig = plt.figure(figsize=(10, 10))
        gs = plt.GridSpec(2, 1, height_ratios=(1, 3))
        axf = plt.subplot(gs[0])
        axima = plt.subplot(gs[1], sharex=axf)

        axima.pcolormesh(f, t, ima / detl * 3, vmax=3, vmin=0.3)

        for il, line in enumerate(ima / detl * 3):
            axf.plot(f, line, lw=0.2, ls='-', c='grey', alpha=0.5, label=f"{il}")

        mean_line = np.mean(ima, axis=0)
        axf.plot(f, mean_line / sum_detl * 3, lw=1, c='k',
                 zorder=10, label="mean", ls='-')

        axima.set_xlabel("Frequency")
        axima.set_ylabel("Time")
        axf.set_ylabel(r"Significance ($\sigma$)")
        axf.set_xlim([results.f0, results.f1])

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        all_images.append(image)
    if HAS_IMAGEIO:
        imageio.mimsave(gif_name, all_images, fps=1)

    return all_images


@njit(parallel=True, nogil=True)
def _fast_step(profiles, L, Q, linbinshifts, quabinshifts, nbin, n=2):
    twopiphases = 2 * np.pi * np.arange(0, 1, 1 / nbin)
    stats = np.zeros_like(L)
    repeated_profiles = np.hstack((profiles, profiles, profiles))

    for i in prange(len(linbinshifts)):
        # This zeros needs to be here, not outside the parallel loop, or
        # the threads will try to write it all at the same time
        newprof = np.zeros(profiles.shape)
        for j in range(len(quabinshifts)):
            newprof = shift_and_select(repeated_profiles, L[i, j], Q[i, j],
                                       newprof)
            splat_prof = np.sum(newprof, axis=0)
            local_stat = z_n_fast(twopiphases, norm=splat_prof, n=n)
            # local_stat = stat(splat_prof)
            stats[i, j] = local_stat

    return stats


@njit(parallel=True)
def _fast_phase_fdot(ts, mean_f, mean_fdot=0):
    phases = ts * mean_f + 0.5 * ts * ts * mean_fdot
    return phases - np.floor(phases)


@njit(parallel=True)
def _fast_phase(ts, mean_f):
    phases = ts * mean_f
    return phases - np.floor(phases)


def search_with_qffa_step(
        times: np.double,
        mean_f: np.double,
        mean_fdot=0,
        nbin=16,
        nprof=64,
        npfact=2,
        oversample=8,
        n=1,
        search_fdot=True):
    """Single step of quasi-fast folding algorithm."""

    # Cast to standard double, or the fast_histogram.histogram2d will fail
    # horribly.

    phases = _fast_phase_fdot(times, mean_f, mean_fdot)

    profiles = histogram2d(phases, times, range=[
        [0, 1], [times[0], times[-1]]], bins=(nbin, nprof)).T

    # Assume times are sorted
    t1, t0 = times[-1], times[0]

    # dn = max(1, int(nbin / oversample))
    linbinshifts = np.linspace(-nbin * npfact, nbin * npfact,
                               oversample * npfact)
    if search_fdot:
        quabinshifts = np.linspace(-nbin * npfact, nbin * npfact,
                                   oversample * npfact)
    else:
        quabinshifts = [0]

    dphi = 1 / nbin
    delta_t = (t1 - t0) / 2
    bin_to_frequency = dphi / delta_t
    bin_to_fdot = 2 * dphi / delta_t ** 2

    L, Q = np.meshgrid(linbinshifts, quabinshifts, indexing='ij')

    stats = _fast_step(profiles, L, Q, linbinshifts, quabinshifts, nbin, n=n)

    return L * bin_to_frequency + mean_f, Q * bin_to_fdot + mean_fdot, stats


def search_with_qffa(times, f0, f1, fdot=0, nbin=16, nprof=None, npfact=2,
                     oversample=8, n=1, search_fdot=True, t0=None, t1=None):
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
    log.info(
        f"Maximum error on the phase expected when casting to double: {maxerr}")
    if maxerr > 1 / nbin / 10:
        warnings.warn(
            "Casting to double produces non-negligible phase errors. "
            "Please use shorter light curves.",
            AstropyUserWarning)

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

    for ii, i in enumerate(show_progress(allvalues)):
        offset = step * i
        fdot_offset = 0

        mean_f = np.double(frequency + offset + 0.12 * step)
        mean_fdot = np.double(fdot + fdot_offset)
        fgrid, fdotgrid, stats = \
            search_with_qffa_step(times, mean_f, mean_fdot=mean_fdot,
                                  nbin=nbin, nprof=nprof, npfact=npfact,
                                  oversample=oversample, n=n,
                                  search_fdot=search_fdot)

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
    return all_fgrid.T, all_fdotgrid.T, all_stats.T, step, fdotstep, length


def folding_search(events, fmin, fmax, step=None,
                   func=epoch_folding_search, oversample=2, fdotmin=0,
                   fdotmax=0, fdotstep=None, expocorr=False, **kwargs):

    times = (events.time - events.gti[0, 0]).astype(np.float64)
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
        print("Searching {} frequencies and {} fdots".format(len(trial_freqs),
                                                             len(trial_fdots)))
    else:
        print("Searching {} frequencies".format(len(trial_freqs)))

    results = func(times, trial_freqs, fdots=trial_fdots,
                   expocorr=expocorr, gti=gti, **kwargs)
    if len(results) == 2:
        frequencies, stats = results
        return frequencies, stats, step, length
    elif len(results) == 3:
        frequencies, fdots, stats = results
        return frequencies, fdots, stats, step, fdotstep, length


def dyn_folding_search(events, fmin, fmax, step=None,
                       func=epoch_folding_search, oversample=2,
                       time_step=128, **kwargs):
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
    fig = plt.figure('Dynamical search')
    plt.pcolormesh(frequencies, times, np.array(stats))
    plt.xlabel('Frequency')
    plt.ylabel('Time')
    plt.savefig('Dyn.png')
    plt.close(fig)
    return times, frequencies, np.array(stats)


def _common_parser(args=None):
    from .base import _add_default_args, check_negative_numbers_in_args
    description = ('Search for pulsars using the epoch folding or the Z_n^2 '
                   'algorithm')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("-f", "--fmin", type=float, required=True,
                        help="Minimum frequency to fold")
    parser.add_argument("-F", "--fmax", type=float, required=True,
                        help="Maximum frequency to fold")
    parser.add_argument("--emin", default=None, type=int,
                        help="Minimum energy (or PI if uncalibrated) to plot")
    parser.add_argument("--emax", default=None, type=int,
                        help="Maximum energy (or PI if uncalibrated) to plot")
    parser.add_argument("--fdotmin", type=float, required=False,
                        help="Minimum fdot to fold", default=0)
    parser.add_argument("--fdotmax", type=float, required=False,
                        help="Maximum fdot to fold", default=0)
    parser.add_argument("--dynstep", type=int, required=False,
                        help="Dynamical EF step", default=128)
    parser.add_argument('-n', "--nbin", default=128, type=int,
                        help="Number of phase bins of the profile")
    parser.add_argument("--segment-size", default=1e32, type=float,
                        help="Size of the event list segment to use (default "
                             "None, implying the whole observation)")
    parser.add_argument("--step", default=None, type=float,
                        help="Step size of the frequency axis. Defaults to "
                             "1/oversample/observ.length. ")
    parser.add_argument("--oversample", default=None, type=float,
                        help="Oversampling factor - frequency resolution "
                             "improvement w.r.t. the standard FFT's "
                             "1/observ.length.")
    parser.add_argument("--fast",
                        help="Use a faster folding algorithm. "
                             "It automatically searches for the first spin "
                             "derivative using an optimized step."
                             "This option ignores expocorr, fdotmin/max, "
                             "segment-size, and step",
                        default=False, action='store_true')
    parser.add_argument("--transient",
                        help="Look for transient emission (produces an animated"
                             " GIF with the dynamic Z search)",
                        default=False, action='store_true')
    parser.add_argument("--expocorr",
                        help="Correct for the exposure of the profile bins. "
                             "This method is *much* slower, but it is useful "
                             "for very slow pulsars, where data gaps due to "
                             "occultation or SAA passages can significantly "
                             "alter the exposure of different profile bins.",
                        default=False, action='store_true')

    parser.add_argument("--find-candidates",
                        help="Find pulsation candidates using thresholding",
                        default=False, action='store_true')
    parser.add_argument("--conflevel", default=99, type=float,
                        help="percent confidence level for thresholding "
                             "[0-100).")

    parser.add_argument("--fit-candidates",
                        help="Fit the candidate peaks in the periodogram",
                        default=False, action='store_true')
    parser.add_argument("--curve", default='sinc', type=str,
                        help="Kind of curve to use (sinc or Gaussian)")
    parser.add_argument("--fit-frequency", type=float,
                        help="Force the candidate frequency to FIT_FREQUENCY")

    # Only relevant to z search
    parser.add_argument('-N', default=2, type=int,
                        help="The number of harmonics to use in the search "
                             "(the 'N' in Z^2_N; only relevant to Z search!)")

    args = check_negative_numbers_in_args(args)
    _add_default_args(parser, ['deorbit', 'loglevel', 'debug'])

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'

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
        raise ValueError('The fast option is only available for z searches')

    for i_f, fname in enumerate(files):
        kwargs = {}
        baseline = args.nbin
        kind = 'EF'
        n = 1
        if func == z_n_search:
            n = args.N
            kwargs = {'nharm': args.N}
            baseline = args.N
            kind = 'Z2n'
        events = load_events(fname)
        mjdref = events.mjdref
        if args.emin is not None or args.emax is not None:
            events, elabel = filter_energy(events, args.emin, args.emax)

        if args.deorbit_par is not None:
            events = deorbit_events(events, args.deorbit_par)

        if args.fast:
            oversample = assign_value_if_none(args.oversample, 8)

        else:
            oversample = assign_value_if_none(args.oversample, 2)

        if args.transient:
            results = transient_search(events.time, args.fmin, args.fmax, fdot=0,
                                       nbin=args.nbin, n=n,
                                       nprof=None, oversample=oversample)
            plot_transient_search(results, hen_root(fname) + '_transient.gif')
            continue

        if not args.fast:
            results = \
                folding_search(events, args.fmin, args.fmax, step=args.step,
                               func=func,
                               oversample=oversample, nbin=args.nbin,
                               expocorr=args.expocorr, fdotmin=args.fdotmin,
                               fdotmax=args.fdotmax,
                               segment_size=args.segment_size, **kwargs)
            ref_time = (events.gti[0, 0])
        else:
            results = \
                search_with_qffa(events.time, args.fmin, args.fmax, fdot=0,
                                 nbin=args.nbin, n=n,
                                 nprof=None, npfact=2, oversample=oversample)
            ref_time = (events.time[-1] + events.time[0]) / 2

        length = events.time.max() - events.time.min()
        segment_size = np.min([length, args.segment_size])
        M = length // segment_size

        fdots = 0
        if len(results) == 4:
            frequencies, stats, step, length = results
        elif len(results) == 6:
            frequencies, fdots, stats, step, fdotsteps, length = results

        if length > args.dynstep and not args.fast:
            _ = dyn_folding_search(events, args.fmin, args.fmax, step=step,
                                   func=func, oversample=oversample,
                                   time_step=args.dynstep, **kwargs)

        efperiodogram = EFPeriodogram(frequencies, stats, kind, args.nbin,
                                      args.N, fdots=fdots, M=M,
                                      segment_size=segment_size,
                                      filename=fname, parfile=args.deorbit_par,
                                      emin=args.emin, emax=args.emax,
                                      mjdref=mjdref,
                                      pepoch=mjdref + ref_time / 86400)

        if args.find_candidates:
            threshold = 1 - args.conflevel / 100
            best_peaks, best_stat = \
                search_best_peaks(frequencies, stats, threshold)
            efperiodogram.peaks = best_peaks
            efperiodogram.peak_stat = best_stat
        elif args.fit_frequency is not None:
            efperiodogram.peaks = best_peaks
            efperiodogram.peak_stat = [0]

        best_models = []

        if args.fit_candidates:
            search_width = 5 * oversample * step
            for f in best_peaks:
                good = np.abs(frequencies - f) < search_width
                if args.curve.lower() == 'sinc':
                    best_fun = fit(frequencies[good], stats[good], f,
                                   obs_length=length, baseline=baseline)
                elif args.curve.lower() == 'gaussian':
                    best_fun = fit(frequencies[good], stats[good], f,
                                   baseline=baseline)
                else:
                    raise ValueError('`--curve` arg must be sinc or gaussian')

                best_models.append(best_fun)

        efperiodogram.best_fits = best_models

        out_fname = hen_root(fname) + '_{}'.format(kind)
        if args.emin is not None or args.emax is not None:
            emin = assign_value_if_none(args.emin, '**')
            emax = assign_value_if_none(args.emax, '**')
            out_fname += f'_{emin}-{emax}keV'

        save_folding(efperiodogram,
                     out_fname + HEN_FILE_EXTENSION)


def main_efsearch(args=None):
    """Main function called by the `HENefsearch` command line script."""

    with log.log_to_file('HENefsearch.log'):
        _common_main(args, epoch_folding_search)


def main_zsearch(args=None):
    """Main function called by the `HENzsearch` command line script."""

    with log.log_to_file('HENzsearch.log'):
        _common_main(args, z_n_search)
