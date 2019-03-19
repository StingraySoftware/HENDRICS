# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Search for pulsars."""
from __future__ import (absolute_import, division, print_function)

from .io import load_events, EFPeriodogram, save_folding, load_folding, \
    HEN_FILE_EXTENSION
from .base import hen_root
from stingray.pulse.search import epoch_folding_search, z_n_search, \
    search_best_peaks, phaseogram
from stingray.gti import time_intervals_from_gtis
from stingray.utils import assign_value_if_none
from stingray.pulse.modeling import fit_sinc, fit_gaussian

import numpy as np
import os
import logging
import argparse
from .base import deorbit_events


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


def decide_binary_parameters(length, freq_range, porb_range, asini_range,
                             fdot_range=[0, 0], NMAX=10,
                             csv_file='db.csv', reset=False):
    import pandas as pd
    count = 0
    omega_range = [1 / porb_range[1], 1 / porb_range[0]]
    columns = ['freq', 'fdot', 'X', 'Porb', 'done', 'max_stat', 'min_stat', 'best_T0']

    df = 1 / length
    print('Recommended frequency steps: {}'.format(int(np.diff(freq_range)[0] // df + 1)))
    while count < NMAX:
        # In any case, only the first loop deletes the file
        if count > 0:
            reset = False
        block_of_data = []
        freq = np.random.uniform(freq_range[0], freq_range[1])
        fdot = np.random.uniform(fdot_range[0], fdot_range[1])

        dX = 1/(TWOPI * freq)

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
        except:
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
            chunk['max_stat'][i] = max_stats
            chunk['min_stat'][i] = min_stats
            chunk['best_T0'][i] = best_T0

            chunk['done'][i] = True
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
        except:
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
    description = ('Search for pulsars using the epoch folding or the Z_n^2 '
                   'algorithm')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("-f", "--fmin", type=float, required=True,
                        help="Minimum frequency to fold")
    parser.add_argument("-F", "--fmax", type=float, required=True,
                        help="Maximum frequency to fold")
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
    parser.add_argument("--oversample", default=2, type=float,
                        help="Oversampling factor - frequency resolution "
                             "improvement w.r.t. the standard FFT's "
                             "1/observ.length.")
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

    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    # Only relevant to z search
    parser.add_argument('-N', default=2, type=int,
                        help="The number of harmonics to use in the search "
                             "(the 'N' in Z^2_N; only relevant to Z search!)")
    parser.add_argument("--deorbit-par",
                        help=("Deorbit data with this parameter file (requires PINT installed)"),
                        default=None,
                        type=str)

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='HENefsearch.log', level=numeric_level,
                        filemode='w')

    return args


def _common_main(args, func):
    args = _common_parser(args)
    files = args.files
    if args.fit_candidates and args.fit_frequency is None:
        args.find_candidates = True
    elif args.fit_candidates and args.fit_frequency is not None:
        args.find_candidates = False
        best_peaks = [args.fit_frequency]

    for i_f, fname in enumerate(files):
        kwargs = {}
        baseline = args.nbin
        kind = 'EF'
        if func == z_n_search:
            kwargs = {'nharm': args.N}
            baseline = args.N
            kind = 'Z2n'
        events = load_events(fname)
        if args.deorbit_par is not None:
            events = deorbit_events(events, args.deorbit_par)

        results = \
            folding_search(events, args.fmin, args.fmax, step=args.step,
                           func=func,
                           oversample=args.oversample, nbin=args.nbin,
                           expocorr=args.expocorr, fdotmin=args.fdotmin,
                           fdotmax=args.fdotmax,
                           segment_size=args.segment_size, **kwargs)
        length = events.time.max() - events.time.min()
        segment_size = np.min([length, args.segment_size])
        M = length // segment_size

        fdots = 0
        if len(results) == 4:
            frequencies, stats, step, length = results
        elif len(results) == 6:
            frequencies, fdots, stats, step, fdotsteps, length = results

        if length > args.dynstep:
            _ = dyn_folding_search(events, args.fmin, args.fmax, step=step,
                                   func=func, oversample=args.oversample,
                                   time_step=args.dynstep, **kwargs)

        efperiodogram = EFPeriodogram(frequencies, stats, kind, args.nbin,
                                      args.N, fdots=fdots, M=M,
                                      segment_size=segment_size,
                                      filename=fname, parfile=args.deorbit_par)

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
            search_width = 5 * args.oversample * step
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

        save_folding(efperiodogram,
                     hen_root(fname) + '_{}'.format(kind) + HEN_FILE_EXTENSION)


def main_efsearch(args=None):
    """Main function called by the `HENefsearch` command line script."""
    _common_main(args, epoch_folding_search)


def main_zsearch(args=None):
    """Main function called by the `HENzsearch` command line script."""
    _common_main(args, z_n_search)
