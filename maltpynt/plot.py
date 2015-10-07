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
from .base import create_gti_mask, _assign_value_if_none
from .base import detection_level
from .fspec import rms_normalize_pds
import logging
import numpy as np


def _baseline_fun(x, a):
    """A constant function."""
    return a


def _value_or_none(dict_like, key):
    try:
        return dict_like[key]
    except:
        return None


def plot_generic(fnames, vars, errs=None, figname=None, xlog=None, ylog=None):
    """Generic plotting function."""
    if is_string(fnames):
        fnames = [fnames]
    figname = _assign_value_if_none(figname,
                                    '{0} vs {1}'.format(vars[1], vars[0]))
    plt.figure(figname)
    ax = plt.gca()
    if xlog:
        ax.set_xscale('log', nonposx='clip')
    if ylog:
        ax.set_yscale('log', nonposy='clip')
    rainbow = ax._get_lines.color_cycle
    xlabel, ylabel = vars
    xlabel_err, ylabel_err = None, None
    if errs is not None:
        xlabel_err, ylabel_err = errs

    for i, fname in enumerate(fnames):
        data = load_data(fname)
        color = next(rainbow)
        xdata = data[xlabel]
        ydata = data[ylabel]
        xdata_err = _value_or_none(data, xlabel_err)
        ydata_err = _value_or_none(data, ylabel_err)
        plt.errorbar(xdata, ydata, yerr=ydata_err, xerr=xdata_err, fmt='-',
                     drawstyle='steps-mid', color=color, label=fname)

    plt.xlabel(vars[0])
    plt.ylabel(vars[1])
    plt.legend()


def plot_pds(fnames, figname=None, xlog=None, ylog=None):
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
            xlog = _assign_value_if_none(xlog, False)
            ylog = _assign_value_if_none(ylog, False)
        except:
            flo = pdsdata['flo']
            fhi = pdsdata['fhi']
            freq = (fhi + flo) / 2
            xlog = _assign_value_if_none(xlog, True)
            ylog = _assign_value_if_none(ylog, True)

        pds = pdsdata['pds']
        epds = pdsdata['epds']
        npds = pdsdata['npds']
        norm = pdsdata['norm']
        rebin = pdsdata['rebin']
        ctrate = pdsdata['ctrate']
        back_ctrate = pdsdata['back_ctrate']

        nbin = len(pds[1:])

        lev = detection_level(nbin, n_summed_spectra=npds, n_rebin=rebin)
        if norm == "rms":
            lev, _ = rms_normalize_pds(lev, 0,
                                       source_ctrate=ctrate,
                                       back_ctrate=back_ctrate)

        color = next(rainbow)

        p, pcov = curve_fit(_baseline_fun, freq, pds, p0=[2], sigma=epds)
        logging.info('White noise level is {0}'.format(p[0]))
        pds -= p[0]
        if xlog and ylog:
            plt.figure('PDS - Loglog')
        else:
            plt.figure('PDS')
        ax = plt.gca()
        if xlog:
            ax.set_xscale('log', nonposx='clip')
        if ylog:
            ax.set_yscale('log', nonposy='clip')

        level = lev - p[0]
        if norm == 'Leahy' or (norm == 'rms' and (not xlog or not ylog)):
            plt.errorbar(freq[1:], pds[1:], yerr=epds[1:], fmt='-',
                         drawstyle='steps-mid', color=color, label=fname)
        elif norm == 'rms' and xlog and ylog:
            plt.errorbar(freq[1:], pds[1:] * freq[1:],
                         yerr=epds[1:] * freq[1:], fmt='-',
                         drawstyle='steps-mid', color=color, label=fname)
            level *= freq

        if np.any(level < 0):
            continue
        if isinstance(level, collections.Iterable):
            plt.plot(freq, level, ls='--', color=color)
        else:
            plt.axhline(level, ls='--', color=color)

    plt.xlabel('Frequency')
    if norm == 'rms':
        plt.ylabel('(rms/mean)^2')
    elif norm == 'Leahy':
        plt.ylabel('Leahy power')
    plt.legend()

    if figname is not None:
        plt.savefig(figname)


def plot_cospectrum(fnames, figname=None, xlog=None, ylog=None):
    """Plot the cospectra from a list of CPDSs, or a single one."""
    if is_string(fnames):
        fnames = [fnames]
    for fname in fnames:
        pdsdata = load_pds(fname)

        try:
            freq = pdsdata['freq']
            xlog = _assign_value_if_none(xlog, False)
            ylog = _assign_value_if_none(ylog, False)
        except:
            flo = pdsdata['flo']
            fhi = pdsdata['fhi']
            freq = (fhi + flo) / 2
            xlog = _assign_value_if_none(xlog, True)
            ylog = _assign_value_if_none(ylog, True)

        cpds = pdsdata['cpds']

        cospectrum = cpds.real
        if xlog and ylog:
            plt.figure('Cospectrum - Loglog')
        else:
            plt.figure('Cospectrum')
        ax = plt.gca()
        if xlog:
            ax.set_xscale('log', nonposx='clip')
        if ylog:
            ax.set_yscale('log', nonposy='clip')

        plt.xlabel('Frequency')
        if xlog and ylog:
            plt.plot(freq[1:], freq[1:] * cospectrum[1:],
                     drawstyle='steps-mid', label=fname)
            plt.ylabel('Cospectrum * Frequency')
        else:
            plt.plot(freq[1:], cospectrum[1:], drawstyle='steps-mid',
                     label=fname)

            plt.ylabel('Cospectrum')
    plt.legend()

    if figname is not None:
        plt.savefig(figname)


def plot_lc(lcfiles, figname=None, fromstart=False, xlog=None, ylog=None):
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
        if fromstart:
            time -= lcdata['Tstart']
            gti -= lcdata['Tstart']

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
    """Main function called by the `MPplot` command line script."""
    import argparse

    description = \
        'Plot the content of MaLTPyNT light curves and frequency spectra'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("--noplot", help="Only create images, do not plot",
                        default=False, action='store_true')
    parser.add_argument("--xlog", help="Use logarithmic X axis",
                        default=None, action='store_true')
    parser.add_argument("--ylog", help="Use logarithmic Y axis",
                        default=None, action='store_true')
    parser.add_argument("--xlin", help="Use linear X axis",
                        default=None, action='store_true')
    parser.add_argument("--ylin", help="Use linear Y axis",
                        default=None, action='store_true')
    parser.add_argument("--fromstart",
                        help="Times are measured from the start of the "
                             "observation (only relevant for light curves)",
                        default=False, action='store_true')
    parser.add_argument("--axes", nargs=2, type=str,
                        help="Plot two variables contained in the file",
                        default=None)

    args = parser.parse_args(args)
    if args.xlin is not None:
        args.xlog = False
    if args.ylin is not None:
        args.ylog = False
    for fname in args.files:
        ftype, contents = get_file_type(fname)
        if args.axes is not None:
            plot_generic(fname, args.axes, xlog=args.xlog, ylog=args.ylog)
            continue
        if ftype == 'lc':
            plot_lc(fname, fromstart=args.fromstart, xlog=args.xlog,
                    ylog=args.ylog)
        elif ftype[-4:] == 'cpds':
            plot_cospectrum(fname, xlog=args.xlog, ylog=args.ylog)
        elif ftype[-3:] == 'pds':
            plot_pds(fname, xlog=args.xlog, ylog=args.ylog)

    if not args.noplot:
        plt.show()
