"""Interactive phaseogram."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .io import load_events, load_folding
from stingray.pulse.pulsar import fold_events, pulse_phase
from stingray.utils import assign_value_if_none

import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
import six
from abc import ABCMeta, abstractmethod
from matplotlib.widgets import Slider, Button
import astropy.units as u


def _check_odd(n):
    return n // 2 * 2 + 1


def run_folding(file, freq, fdot=0, fddot=0, nbin=16, nebin=16, tref=0,
                test=False, emin=0, emax=1e32, norm='to1',
                smooth_window=None, **opts):

    file_label = ''
    ev = load_events(file)
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
        gs = GridSpec(2, 1, height_ratios=(1, 3))
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
    else:
        ax0 = plt.subplot()

    # Plot pulse profile
    max = np.max(smooth)
    min = np.min(smooth)
    ax0.plot(meanbins, profile, drawstyle='steps-mid',
             color='grey')
    ax0.plot(meanbins, smooth, drawstyle='steps-mid',
             label='Smooth profile '
                   '(P.F. = {:.1f}%)'.format(100 * (max - min) / max),
             color='k')
    ax0.axhline(max, lw=1, color='k')
    ax0.axhline(min, lw=1, color='k')


    mean = np.mean(profile)
    ax0.fill_between(meanbins, mean - np.sqrt(mean), mean + np.sqrt(mean))
    ax0.axhline(mean, ls='--')
    ax0.legend()

    if plot_energy:
        ax1.pcolormesh(X, Y, hist2d.T)
        ax1.semilogy()

        ax1.set_xlabel('Phase')
        ax1.set_ylabel(elabel)

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
                        default=0)
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
                norm=args.norm)