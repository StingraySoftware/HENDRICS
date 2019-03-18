"""Interactive phaseogram."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .io import load_events
from stingray.pulse.pulsar import fold_events, pulse_phase, get_TOA
from stingray.utils import assign_value_if_none

import numpy as np
import logging
import argparse
from scipy.signal import savgol_filter
from scipy import optimize
from astropy.stats import poisson_conf_interval

import copy
try:
    from tqdm import tqdm as show_progress
except ImportError:
    def show_progress(a):
        return a

try:
    import pint.toa as toa
    import pint
    HAS_PINT = True
except ImportError:
    HAS_PINT = False
from .base import deorbit_events


def _load_and_prepare_TOAs(mjds, errs_us=None, ephem="DE405"):
    errs_us = assign_value_if_none(errs_us, np.zeros_like(mjds))

    toalist = [None] * len(mjds)
    for i, m in enumerate(mjds):
        toalist[i] = \
            toa.TOA(m, error=errs_us[i], obs='Barycenter', scale='tdb')

    toalist = toa.TOAs(toalist = toalist)
    if 'tdb' not in toalist.table.colnames:
        toalist.compute_TDBs()
    if 'ssb_obs_pos' not in toalist.table.colnames:
        toalist.compute_posvels(ephem, False)
    return toalist


def get_TOAs_from_events(events, folding_length, *frequency_derivatives,
                         **kwargs):
    """Get TOAs of pulsation.

    Parameters
    ----------
    events : array-like
        event arrival times
    folding_length : float
        length of sub-intervals to fold
    *frequency_derivatives : floats
        pulse frequency, first derivative, second derivative, etc.

    Other parameters
    ----------------
    pepoch : float, default None
        Epoch of timing solution, in the same units as ev_times. If none, the
        first event time is used.
    mjdref : float, default None
        Reference MJD
    template : array-like, default None
        The pulse template
    nbin : int, default 16
        The number of bins in the profile (overridden by the dimension of the
        template)
    timfile : str, default 'out.tim'
        file to save the TOAs to (if PINT is installed)
    gti: [[g0_0, g0_1], [g1_0, g1_1], ...]
         Good time intervals. Defaults to None
    quick: bool
         If True, use a quicker fitting algorithms for TOAs. Defaults to False
    position: `astropy.SkyCoord` object
         Position of the object

    Returns
    -------
    toas : array-like
        list of times of arrival. If ``mjdref`` is specified, they are
        expressed as MJDs, otherwise in MET
    toa_err : array-like
        errorbars on TOAs, in the same units as TOAs.
    """
    import matplotlib.pyplot as plt
    template = kwargs['template'] if 'template' in kwargs else None
    mjdref = kwargs['mjdref'] if 'mjdref' in kwargs else None
    nbin = kwargs['nbin'] if 'nbin' in kwargs else 16
    pepoch = kwargs['pepoch'] if 'pepoch' in kwargs else None
    timfile = kwargs['timfile'] if 'timfile' in kwargs else 'out.tim'
    gti = kwargs['gti'] if 'gti' in kwargs else None
    label = kwargs['label'] if 'label' in kwargs else None
    quick = kwargs['quick'] if 'quick' in kwargs else False
    position = kwargs['position'] if 'position' in kwargs else None

    pepoch = assign_value_if_none(pepoch, events[0])
    gti = assign_value_if_none(gti, [[events[0], events[-1]]])
    # run exposure correction only if there are less than 1000 pulsations
    # in the interval
    length = gti.max() - gti.min()
    expocorr = folding_length < (1000 / frequency_derivatives[0])

    if template is not None:
        nbin = len(template)
        additional_phase = np.argmax(template) / nbin
    else:
        phase, profile, profile_err = \
            fold_events(copy.deepcopy(events), *frequency_derivatives,
                        ref_time=pepoch, gtis=copy.deepcopy(gti),
                        expocorr=expocorr, nbin=nbin)
        fit_pars_save, _, _ = \
            fit_profile_with_sinusoids(profile, profile_err, nperiods=1,
                                       baseline=True)
        template = std_fold_fit_func(fit_pars_save, phase)
        fig = plt.figure()
        plt.plot(phase, profile, drawstyle='steps-mid')
        plt.plot(phase, template, drawstyle='steps-mid')
        plt.savefig(timfile.replace('.tim', '') + '.png')
        plt.close(fig)
        # start template from highest bin!
        # template = np.roll(template, -np.argmax(template))
        template *= folding_length / length
        template_fine = std_fold_fit_func(fit_pars_save,
                                          np.arange(0, 1, 0.001))
        additional_phase = np.argmax(template_fine) / len(template_fine)

    starts = np.arange(gti[0, 0], gti[-1, 1], folding_length)

    toas = []
    toa_errs = []
    for start in show_progress(starts):
        stop = start + folding_length
        good = (events >= start) & (events < stop)
        events_tofold = events[good]
        if len(events_tofold) < nbin:
            continue
        gtis_tofold = \
            copy.deepcopy(gti[(gti[:, 0] < stop) & (gti[:, 1] > start)])
        gtis_tofold[0, 0] = start
        gtis_tofold[-1, 1] = stop

        local_f = frequency_derivatives[0]
        for i_f, f in enumerate(frequency_derivatives[1:]):
            local_f += 1 / np.math.factorial(i_f + 1) * (start - pepoch) ** (i_f + 1) * f

        fder = copy.deepcopy(list(frequency_derivatives))
        fder[0] = local_f
        phase, profile, profile_err = \
            fold_events(events_tofold, *fder,
                        ref_time=start, gtis=gtis_tofold,
                        expocorr=expocorr, nbin=nbin)

        # BAD!BAD!BAD!
        # [[Pay attention to time reference here.
        # We are folding wrt pepoch, and calculating TOAs wrt start]]

        toa, toaerr = \
            get_TOA(profile, 1/frequency_derivatives[0], start,
                    template=template, additional_phase=additional_phase,
                    quick=quick, debug=True)
        toas.append(toa)
        toa_errs.append(toaerr)

    toas, toa_errs = np.array(toas), np.array(toa_errs)

    if mjdref is not None:
        toas = toas / 86400 + mjdref
        toa_errs = toa_errs * 1e6
        if HAS_PINT:
            label = assign_value_if_none(label, 'hendrics')
            toa_list = _load_and_prepare_TOAs(toas, errs_us=toa_errs)
            # workaround until PR #368 is accepted in pint
            toa_list.table['clkcorr'] = 0
            toa_list.write_TOA_file(timfile, name=label, format='Tempo2')

        print('TOA(MJD)  TOAerr(us)')
    else:
        print('TOA(MET)  TOAerr(us)')

    for t, e in zip(toas, toa_errs):
        print(t, e)

    return toas, toa_errs


def _check_odd(n):
    return n // 2 * 2 + 1


def dbl_cos_fit_func(p, x):
    # the frequency is fixed
    '''
    A double sinus (fundamental + 1st harmonic) used as a fit function
    '''
    startidx = 0
    base = 0
    if len(p) % 2 != 0:
        base = p[0]
        startidx = 1
    first_harm = \
        p[startidx] * np.cos(2*np.pi*x + 2*np.pi*p[startidx + 1])
    second_harm = \
        p[startidx + 2] * np.cos(4.*np.pi*x + 4*np.pi*p[startidx + 3])
    return base + first_harm + second_harm


def std_fold_fit_func(p, x):
    '''Chooses the fit function used in the fit.'''

    return dbl_cos_fit_func(p, x)


def std_residuals(p, x, y):
    '''The residual function used in the fit.'''
    return std_fold_fit_func(p, x) - y


def adjust_amp_phase(pars):
    '''Give the phases in the interval between 0 and 1.
    The calculation is based on the amplitude and phase given as input

    pars[0] is the initial amplitude; pars[1] is the initial phase
    If amplitude is negative, it makes it positive and changes the phase
    accordingly
    '''
    if pars[0] < 0:
        pars[0] = - pars[0]
        pars[1] += 0.5
    if pars[1] < -1:
        pars[1] += np.floor(-pars[1])
    if pars[1] > 1:
        pars[1] -= np.floor(pars[1])
    pars[1] = pars[1] - np.ceil(pars[1])
    return pars


def fit_profile_with_sinusoids(profile, profile_err, debug=False, nperiods=1,
                               baseline=False):
    '''
    Fit a folded profile with the std_fold_fit_func.

    Tries a number of different initial values for the fit, and returns the
    result of the best chi^2 fit

    Parameters
    ----------
    profile : array of floats
        The folded profile
    profile_err : array of floats
        the error on the folded profile elements

    Other parameters
    ----------------
    debug : bool, optional
        print debug info
    nperiods : int, optional
        number of periods in the folded profile. Default 1.

    Returns
    -------
    fit_pars : array-like
        the best-fit parameters
    success : bool
        whether the fit succeeded or not
    chisq : float
        the best chi^2
    '''
    x = np.arange(0, len(profile) * nperiods, nperiods) / float(len(profile))
    guess_pars = [max(profile) - np.mean(profile),
                  x[np.argmax(profile[:len(profile) // nperiods])] - 0.25,
                  (max(profile) - np.mean(profile)) / 2., 0.]
    startidx = 0
    if baseline:
        guess_pars = [np.mean(profile)] + guess_pars
        if debug:
            print(guess_pars)
        startidx = 1
    chisq_save = 1e32
    fit_pars_save = guess_pars
    success_save = -1
    if debug:
        import matplotlib.pyplot as plt

        fig = plt.figure('Debug profile')
        plt.errorbar(x, profile, yerr=profile_err, drawstyle='steps-mid')
        plt.plot(x, std_fold_fit_func(guess_pars, x), 'r--')

    for phase in np.arange(0., 1., 0.1):
        guess_pars[3 + startidx] = phase
        logging.debug(guess_pars)
        if debug:
            plt.plot(x, std_fold_fit_func(guess_pars, x), 'r--')
        fit_pars, success = optimize.leastsq(std_residuals, guess_pars[:],
                                             args=(x, profile))
        if debug:
            plt.plot(x, std_fold_fit_func(fit_pars, x), 'g--')
        fit_pars[startidx:startidx + 2] = \
            adjust_amp_phase(fit_pars[startidx:startidx + 2])
        fit_pars[startidx + 2:startidx + 4] = \
            adjust_amp_phase(fit_pars[startidx + 2:startidx + 4])
        chisq = np.sum((profile - std_fold_fit_func(fit_pars, x)) ** 2 /
                       profile_err ** 2) / (len(profile) - (startidx + 4))
        if debug:
            plt.plot(x, std_fold_fit_func(fit_pars, x), 'b--')
        if chisq < chisq_save:
            chisq_save = chisq
            fit_pars_save = fit_pars[:]
            success_save = success

    if debug:
        plt.savefig('debug_fit_profile.png')
        plt.close(fig)
    return fit_pars_save, success_save, chisq_save


def fit_profile(profile, profile_err, debug=False, nperiods=1,
                phaseref='default', baseline=False):
    return fit_profile_with_sinusoids(profile, profile_err, debug=debug,
                                      nperiods=nperiods,
                                      baseline=baseline)


def run_folding(file, freq, fdot=0, fddot=0, nbin=16, nebin=16, tref=None,
                test=False, emin=0, emax=1e32, norm='to1',
                smooth_window=None, deorbit_par=None, **opts):
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt

    file_label = ''
    ev = load_events(file)
    if deorbit_par is not None:
        events = deorbit_events(ev, deorbit_par)

    times = ev.time
    gtis = ev.gti
    plot_energy = True
    if hasattr(ev, 'energy') and ev.energy is not None:
        energy = ev.energy
        elabel = 'Energy'
    elif hasattr(ev, 'pi') and ev.pi is not None:
        energy = ev.pi
        elabel = 'PI'
    else:
        energy = np.ones_like(times)
        elabel = ''
        plot_energy = False

    if tref is None:
        tref = times[0]

    good = (energy > emin) & (energy < emax)
    times = times[good]
    energy = energy[good]
    phases = pulse_phase(times - tref, freq, fdot, fddot, to_1=True)

    binx = np.linspace(0, 1, nbin + 1)
    if plot_energy:
        biny = np.percentile(energy, np.linspace(0, 100, nebin + 1))
        biny[0] = emin
        biny[-1] = emax

    profile, _ = np.histogram(phases, bins=binx)
    if smooth_window is None:
        smooth_window = np.min([len(profile), np.max([len(profile) // 10, 5])])
        smooth_window = _check_odd(smooth_window)

    smoothed_profile = savgol_filter(profile, window_length=smooth_window,
                                     polyorder=2, mode='wrap')

    profile = np.concatenate((profile, profile))
    smooth = np.concatenate((smoothed_profile, smoothed_profile))

    if plot_energy:
        histen, _ = np.histogram(energy, bins=biny)

        hist2d, _, _ = np.histogram2d(phases.astype(np.float64),
                                      energy, bins=(binx, biny))

    binx = np.concatenate((binx[:-1], binx + 1))
    meanbins = (binx[:-1] + binx[1:])/2

    if plot_energy:
        hist2d = np.vstack((hist2d, hist2d))
        hist2d_save = np.copy(hist2d)
        X, Y = np.meshgrid(binx, biny)

        if norm == 'ratios':
            hist2d /= smooth[:, np.newaxis]
            hist2d *= histen[np.newaxis, :]
            file_label = '_ratios'
        else:
            hist2d /= histen[np.newaxis, :]
            factor = np.max(hist2d, axis=0)[np.newaxis, :]
            hist2d /= factor
            file_label = '_to1'

    plt.figure()
    if plot_energy:
        gs = GridSpec(2, 2, height_ratios=(1, 3))
        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[1, 0], sharex=ax0)
        ax2 = plt.subplot(gs[1, 1], sharex=ax0)
        ax3 = plt.subplot(gs[0, 1])

    else:
        ax0 = plt.subplot()

    # Plot pulse profile
    max = np.max(smooth)
    min = np.min(smooth)
    ax0.plot(meanbins, profile, drawstyle='steps-mid',
             color='white', zorder=2)
    ax0.plot(meanbins, smooth, drawstyle='steps-mid',
             label='Smooth profile '
                   '(P.F. = {:.1f}%)'.format(100 * (max - min) / max),
             color='k', zorder=3)
    err_low, err_high = \
        poisson_conf_interval(smooth,
                              interval='frequentist-confidence', sigma=3)

    try:
        ax0.fill_between(meanbins, err_low, err_high, color='grey', zorder=1,
                         alpha=0.5, label='3-sigma confidence',
                         step='mid')
    except AttributeError:
        # MPL < 2
        ax0.fill_between(meanbins, err_low, err_high, color='grey', zorder=1,
                         alpha=0.5, label='3-sigma confidence')

    ax0.axhline(max, lw=1, color='k')
    ax0.axhline(min, lw=1, color='k')

    mean = np.mean(profile)
    ax0.fill_between(meanbins, mean - np.sqrt(mean), mean + np.sqrt(mean),
                     alpha=0.5)
    ax0.axhline(mean, ls='--')
    ax0.legend()
    ax0.set_ylim([0, None])

    if plot_energy:
        ax1.pcolormesh(X, Y, hist2d.T)
        ax1.semilogy()

        ax1.set_xlabel('Phase')
        ax1.set_ylabel(elabel)
        ax1.set_xlim([0, 2])

        pfs = []
        errs = []
        meannrgs = (biny[:-1] + biny[1:]) / 2
        for i, prof in enumerate(hist2d_save.T):
            smooth = savgol_filter(prof, window_length=smooth_window,
                                   polyorder=2, mode='wrap')
            max = np.max(smooth)
            min = np.min(smooth)
            pf = 100 * (max - min) / max
            ax2.plot(meanbins, prof, drawstyle='steps-mid',
                     label='{}={:.2f}-{:.2f}'.format(elabel, biny[i],
                                                     biny[i+1], pf))
            std = np.max(prof - smooth)
            ax2.set_xlabel('Phase')
            ax2.set_ylabel('Counts')

            pfs.append(pf)
            errs.append(std / max)

        if len(meannrgs) < 6:
            ax2.legend()
        ax2.set_xlim([0, 2])

        ax3.errorbar(meannrgs, pfs, fmt='o', yerr=errs,
                     xerr=(biny[1:] - biny[:-1]) / 2)
        ax3.semilogx()
        ax3.set_xlabel('Energy')
        ax3.set_ylabel('Pulsed fraction')

    plt.savefig('Energyprofile' + file_label + '.png')
    if not test:  # pragma:no cover
        plt.show()


def main_fold(args=None):
    description = ('Plot a folded profile')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("file", help="Input event file", type=str)
    parser.add_argument("-f", "--freq", type=float, required=False,
                        help="Initial frequency to fold", default=None)
    parser.add_argument("--fdot", type=float, required=False,
                        help="Initial fdot", default=0)
    parser.add_argument("--fddot", type=float, required=False,
                        help="Initial fddot", default=0)
    parser.add_argument("--tref", type=float, required=False,
                        help="Reference time (same unit as time array)",
                        default=None)
    parser.add_argument('-n', "--nbin", default=16, type=int,
                        help="Number of phase bins (X axis) of the profile")
    parser.add_argument("--nebin", default=16, type=int,
                        help="Number of energy bins (Y axis) of the profile")
    parser.add_argument("--emin", default=0, type=int,
                        help="Minimum energy (or PI if uncalibrated) to plot")
    parser.add_argument("--emax", default=1e32, type=int,
                        help="Maximum energy (or PI if uncalibrated) to plot")
    parser.add_argument("--norm", default='to1',
                        help="--norm to1: Normalize hist so that the maximum "
                             "at each energy is one. "
                             "--norm ratios: Divide by mean profile")
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')
    parser.add_argument("--test",
                        help="Just a test. Destroys the window immediately",
                        default=False, action='store_true')
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--deorbit-par",
                        help=("Deorbit data with this parameter file (requires PINT installed)"),
                        default=None,
                        type=str)

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='HENfold.log', level=numeric_level,
                        filemode='w')

    frequency = args.freq
    fdot = args.fdot
    fddot = args.fddot

    run_folding(args.file, freq=frequency, fdot=fdot, fddot=fddot,
                nbin=args.nbin, nebin=args.nebin, tref=args.tref,
                test=args.test, emin=args.emin, emax=args.emax,
                norm=args.norm, deorbit_par=args.deorbit_par)


def z2_n_detection_level(epsilon=0.01, n=2, n_summed_spectra=1, ntrial=1):
    """Return the detection level for the Z^2_n statistics.

    See Buccheri et al. (1983), Bendat and Piersol (1971).

    Parameters
    ----------
    n : int, default 2
        The ``n`` in $Z^2_n$
    epsilon : float, default 0.01
        The fractional probability that the signal has been produced by noise

    Other Parameters
    ----------------
    ntrial : int
        The number of trials executed to find this profile
    n_summed_spectra : int
        Number of Z_2^n periodograms that are being averaged

    Returns
    -------
    detlev : float
        The epoch folding statistics corresponding to a probability
        epsilon * 100 % that the signal has been produced by noise
    """

    from scipy import stats
    retlev = stats.chi2.isf(epsilon / ntrial, 2 * int(n_summed_spectra) * n) \
        / (n_summed_spectra)

    return retlev


def z2_n_probability(z2, n=2, ntrial=1, n_summed_spectra=1):
    """Calculate the probability of a certain folded profile, due to noise.

    Parameters
    ----------
    z2 : float
        A Z^2_n statistics value
    n : int, default 2
        The ``n`` in $Z^2_n$

    Other Parameters
    ----------------
    ntrial : int
        The number of trials executed to find this profile
    n_summed_spectra : int
        Number of Z_2^n periodograms that were averaged to obtain z2

    Returns
    -------
    p : float
        The probability that the Z^2_n value has been produced by noise
    """
    if ntrial > 1:
        import warnings
        warnings.warn("Z2_n: The treatment of ntrial is very rough. "
                      "Use with caution")
    from scipy import stats

    epsilon = ntrial * stats.chi2.sf(z2 * n_summed_spectra,
                                     2 * n * n_summed_spectra)
    return epsilon
