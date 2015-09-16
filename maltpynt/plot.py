# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Quicklook plots."""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

try:
    import matplotlib.pyplot as plt
except:
    # Permit to import the module anyway if matplotlib is not present
    pass
from .io import load_data, get_file_type, load_pds
from .io import is_string
from .base import create_gti_mask
from .base import detection_level
import logging
import numpy as np


def _baseline_fun(x, a):
    """A constant function."""
    return a


def plot_pds(fnames, figname=None):
    """Plot a list of PDSs, or a single one."""
    from scipy.optimize import curve_fit
    import collections
    if is_string(fnames):
        fnames = [fnames]
    ax = plt.subplot(1, 1, 1)
    rainbow = ax._get_lines.color_cycle
    for i, fname in enumerate(fnames):
        pdsdata = load_pds(fname)
        try:
            freq = pdsdata['freq']
        except:
            flo = pdsdata['flo']
            fhi = pdsdata['fhi']
            freq = (fhi + flo) / 2
            plt.loglog()

        pds = pdsdata['pds']
        epds = pdsdata['epds']
        npds = pdsdata['npds']
        norm = pdsdata['norm']
        rebin = pdsdata['rebin']

        nbin = len(pds[1:])

        lev = detection_level(nbin, n_summed_spectra=npds, n_rebin=rebin)

        color = next(rainbow)

        p, pcov = curve_fit(_baseline_fun, freq, pds, p0=[2], sigma=epds)
        logging.info('White noise level is {0}'.format(p[0]))
        pds -= p[0]

        plt.errorbar(freq[1:], pds[1:], yerr=epds[1:], fmt='-',
                     drawstyle='steps-mid', color=color, label=fname)

        if np.any(lev - p[0] < 0):
            continue
        if isinstance(lev, collections.Iterable):
            plt.plot(freq, lev - p[0], ls='--', color=color)
        else:
            plt.axhline(lev - p[0], ls='--', color=color)

    plt.xlabel('Frequency')
    if norm == 'rms':
        plt.ylabel('(rms/mean)^2')
    elif norm == 'Leahy':
        plt.ylabel('Leahy power')
    plt.legend()

    if figname is not None:
        plt.savefig(figname)


def plot_cospectrum(fnames, figname=None):
    """Plot the cospectra from a list of CPDSs, or a single one."""
    if is_string(fnames):
        fnames = [fnames]
    for fname in fnames:
        pdsdata = load_pds(fname)

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
        plt.plot(freq[1:], freq[1:] * cospectrum[1:], drawstyle='steps-mid',
                 label=fname)
        plt.figure('Lin')
        plt.plot(freq[1:], cospectrum[1:], drawstyle='steps-mid',
                 label=fname)

    plt.figure('Log')
    plt.xlabel('Frequency')
    plt.ylabel('Cospectrum * Frequency')
    plt.legend()

    plt.figure('Lin')
    plt.axhline(0, lw=3, ls='--', color='k')
    plt.xlabel('Frequency')
    plt.ylabel('Cospectrum')
    plt.legend()

    if figname is not None:
        plt.savefig(figname)


def plot_lc(lcfiles, figname=None):
    """Plot a list of light curve files, or a single one."""
    if is_string(lcfiles):
        lcfiles = [lcfiles]

    for lcfile in lcfiles:
        logging.info('Loading %s...' % lcfile)
        lcdata = load_data(lcfile)

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

        good = create_gti_mask(time, gti)
        plt.plot(time, lc, drawstyle='steps-mid', color='grey')
        plt.plot(time[good], lc[good], drawstyle='steps-mid', label=lcfile)

    plt.xlabel('Time (s)')
    if instr == 'PCA':
        plt.ylabel('light curve (Ct/bin/PCU)')
    else:
        plt.ylabel('light curve (Ct/bin)')

    plt.legend()
    if figname is not None:
        plt.savefig(figname)


def main(args=None):
    import argparse

    description = \
        'Plot the content of MaLTPyNT light curves and frequency spectra'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("--noplot", help="Only create images, do not plot",
                        default=False, action='store_true')

    args = parser.parse_args(args)
    for fname in args.files:
        ftype, contents = get_file_type(fname)
        if ftype == 'lc':
            plot_lc(fname)
        elif ftype[-4:] == 'cpds':
            plot_cospectrum(fname)
        elif ftype[-3:] == 'pds':
            plot_pds(fname)

    if not args.noplot:
        plt.show()
