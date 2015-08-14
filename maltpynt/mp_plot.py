# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Quicklook plots."""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

try:
    import matplotlib.pyplot as plt
except:
    # Permit to import the module anyway if matplotlib is not present
    pass
from .mp_io import mp_load_data
from .mp_io import is_string
from .mp_base import mp_create_gti_mask
from .mp_base import mp_detection_level
import logging


def baseline_fun(x, a):
    """A constant function."""
    return a


def mp_plot_pds(fnames):
    """Plot a list of PDSs, or a single one."""
    from scipy.optimize import curve_fit
    import collections
    if is_string(fnames):
        fnames = [fnames]
    ax = plt.subplot(1, 1, 1)
    rainbow = ax._get_lines.color_cycle
    for i, fname in enumerate(fnames):
        pdsdata = mp_load_data(fname)
        try:
            freq = pdsdata['freq']
        except:
            flo = pdsdata['flo']
            fhi = pdsdata['fhi']
            freq = (fhi + flo) / 2
            plt.semilogx()

        pds = pdsdata['pds']
        epds = pdsdata['epds']
        npds = pdsdata['npds']
        norm = pdsdata['norm']
        rebin = pdsdata['rebin']

        nbin = len(pds[1:])

        lev = mp_detection_level(nbin, n_summed_spectra=npds, n_rebin=rebin)

        color = next(rainbow)

        p, pcov = curve_fit(baseline_fun, freq, pds, p0=[2], sigma=epds)
        logging.info('White noise level is', p[0])
        pds -= p[0]
        if isinstance(lev, collections.Iterable):
            plt.plot(freq, lev - p[0], color=color)
        else:
            plt.axhline(lev - p[0], color=color)

        plt.errorbar(freq[1:], pds[1:], yerr=epds[1:], fmt='-',
                     drawstyle='steps-mid', color=color)

    plt.xlabel('Frequency')
    if norm == 'rms':
        plt.ylabel('(rms/mean)^2')
    elif norm == 'Leahy':
        plt.ylabel('Leahy power')


def mp_plot_cospectrum(fnames):
    """Plot the cospectra from a list of CPDSs, or a single one."""
    if is_string(fnames):
        fnames = [fnames]
    for fname in fnames:
        pdsdata = mp_load_data(fname)

        try:
            freq = pdsdata['freq']
        except:
            flo = pdsdata['flo']
            fhi = pdsdata['fhi']
            freq = (fhi + flo) / 2

        cpds = pdsdata['cpds']

        cospectrum = cpds.real
        plt.figure('Log')
        ax = plt.gca()
        ax.set_xscale('log', nonposx='clip')
        ax.set_yscale('log', nonposy='clip')
        plt.plot(freq[1:], freq[1:] * cospectrum[1:], drawstyle='steps-mid')
        plt.figure('Lin')
        plt.plot(freq[1:], cospectrum[1:], drawstyle='steps-mid')

    plt.figure('Log')
    plt.xlabel('Frequency')
    plt.ylabel('Cospectrum')

    plt.figure('Lin')
    plt.axhline(0, lw=3, ls='--', color='k')
    plt.xlabel('Frequency')
    plt.ylabel('Cospectrum')


def mp_plot_lc(lcfiles):
    """Plot a list of light curve files, or a single one."""
    if is_string(lcfiles):
        lcfiles = [lcfiles]

    for lcfile in lcfiles:
        logging.info('Loading %s...' % lcfile)
        lcdata = mp_load_data(lcfile)

        time = lcdata['time']
        lc = lcdata['lc']
        gti = lcdata['GTI']
        instr = lcdata['Instr']
        if instr == 'PCA':
            # If RXTE, plot per PCU count rate
            npcus = lcdata['nPCUs']
            lc /= npcus

        for g in gti:
            plt.axvline(g[0], ls='-', color='red')
            plt.axvline(g[1], ls='--', color='red')

        good = mp_create_gti_mask(time, gti)
        plt.plot(time, lc, drawstyle='steps-mid', color='grey')
        plt.plot(time[good], lc[good], drawstyle='steps-mid', color='k')

    plt.xlabel('Time (s)')
    if instr == 'PCA':
        plt.ylabel('light curve (Ct/bin/PCU)')
    else:
        plt.ylabel('light curve (Ct/bin)')


if __name__ == "__main__":
    import sys
    import subprocess as sp

    print('Calling script...')

    args = sys.argv[1:]

    sp.check_call(['MPplot'] + args)
