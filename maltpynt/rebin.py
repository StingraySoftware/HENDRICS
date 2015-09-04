# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to rebin light curves and frequency spectra."""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
from .io import get_file_type
from .io import save_data
from .io import MP_FILE_EXTENSION, get_file_extension
import logging


def const_rebin(x, y, factor, yerr=None, normalize=True):
    """Rebin any pair of variables.

    Might be time and counts, or freq and pds.
    Also possible to rebin the error on y.

    Parameters
    ----------
    x : array-like
    y : array-like
    factor : int
        Rebin factor
    yerr : array-like, optional
        Uncertainties of y values (it is assumed that the y are normally
        distributed)

    Returns
    -------
    new_x : array-like
        The rebinned x array
    new_y : array-like
        The rebinned y array
    new_err : array-like
        The rebinned yerr array

    Other Parameters
    ----------------
    normalize : bool
    """
    arr_dtype = y.dtype
    if factor <= 1:
        res = [x, y]
        if yerr is not None:
            res.append(yerr)
        else:
            res.append(np.zeros(len(y), dtype=arr_dtype))
        return res
    factor = np.long(factor)
    nbin = len(y)

    new_nbins = np.int(nbin / factor)

    y_resh = np.reshape(y[:new_nbins * factor], (new_nbins, factor))
    x_resh = np.reshape(x[:new_nbins * factor], (new_nbins, factor))

    new_y = np.sum(y_resh, axis=1)
    new_x = np.sum(x_resh, axis=1) / factor

    if yerr is not None:
        yerr_resh = np.reshape(yerr[:new_nbins * factor], (new_nbins, factor))
        new_yerr = np.sum(yerr_resh ** 2, axis=1)
    else:
        new_yerr = np.zeros(len(new_x), dtype=arr_dtype)

    if normalize:
        return new_x, new_y / factor, np.sqrt(new_yerr) / factor
    else:
        return new_x, new_y, np.sqrt(new_yerr)


def geom_bin(freq, pds, bin_factor=None, pds_err=None, npds=None,
             return_nbins=False):
    """Given a PDS, bin it geometrically.

    Parameters
    ----------
    freq : array-like
    pds : array-like
    bin_factor : float > 1
    pds_err : array-like

    Returns
    -------
    newfreqlo : array-like
        Lower boundaries of the new frequency bins
    newfreqhi : array-like
        Upper boundaries of the new frequency bins
    newpds : array-like
        The rebinned PDS
    newpds_err : array-like
        The uncertainties on the rebinned PDS points (be careful. Check with
        simulations if it works in your case)
    new_nbins : array-like, optional
        The new number of bins averaged in each PDS point. Only returned if
        return_nbins is True

    Other Parameters
    ----------------
    npds : int
    return_nbins : bool

    Notes
    -----
    Some parts of the code are copied from an algorithm in isisscripts.sl

    """
    from numpy import log10

    df = np.diff(freq)
    assert np.max(df) - np.min(df) < 1e-5 * np.max(df), \
        'This only works for not previously rebinned spectra'

    df = freq[1] - freq[0]

    if npds is None:
        npds = 1.
    if pds_err is None:
        pds_err = np.zeros(len(pds))

    if freq[0] < 1e-10:
        freq = freq[1:]
        pds = pds[1:]
        pds_err = pds_err[1:]

    if bin_factor <= 1:
        logging.warning("Bin factor must be > 1!!")
        f0 = freq - df / 2.
        f1 = freq + df / 2.
        retval = [f0, f1, pds, pds_err]
        if return_nbins:
            retval.append(np.ones(len(pds)) * npds)
        return retval

    # Input frequencies are referred to the center of the bin. But from now on
    # I'll be interested in the start and stop of each frequency bin.
    freq = freq - df / 2
    fmin = min(freq)
    fmax = max(freq) + df

    logstep = log10(bin_factor)
#    maximum number of bins
    nmax = np.int((log10(fmax) - log10(fmin)) / logstep + 0.5)

# Low frequency grid
    flo = fmin * 10 ** (np.arange(nmax) * logstep)
    flo = np.append(flo, [fmax])

# Now the clever part: building a histogram of frequencies
    pds_dtype = pds.dtype
    pdse_dtype = pds_err.dtype

    bins = np.digitize(freq.astype(np.double), flo.astype(np.double))
    newpds = np.zeros(nmax, dtype=pds_dtype) - 1
    newpds_err = np.zeros(nmax, dtype=pdse_dtype)
    newfreqlo = np.zeros(nmax)
    new_nbins = np.zeros(nmax, dtype=np.long)
    for i in range(nmax):
        good = bins == i
        ngood = np.count_nonzero(good)
        new_nbins[i] = ngood
        if ngood == 0:
            continue
        newpds[i] = np.sum(pds[good]) / ngood
        newfreqlo[i] = np.min(freq[good])
        newpds_err[i] = np.sqrt(np.sum(pds_err[good] ** 2)) / ngood
    good = new_nbins > 0
    new_nbins = new_nbins[good] * npds
    newfreqlo = newfreqlo[good]
    newpds = newpds[good]
    newpds_err = newpds_err[good]
    newfreqhi = newfreqlo[1:]
    newfreqhi = np.append(newfreqhi, [fmax])

    retval = [newfreqlo, newfreqhi, newpds, newpds_err]
    if return_nbins:
        retval.append(new_nbins)

    return retval


def rebin_file(filename, rebin):
    """Rebin the contents of a file, be it a light curve or a spectrum."""
    ftype, contents = get_file_type(filename)
    do_dyn = False
    if 'dyn{0}'.format(ftype) in contents.keys():
        do_dyn = True

    if ftype == 'lc':
        x = contents['time']
        y = contents['lc']
        ye = np.sqrt(y)
        logging.info('Applying a constant rebinning')
        x, y, ye = \
            const_rebin(x, y, rebin, ye, normalize=False)
        contents['time'] = x
        contents['lc'] = y
        if 'rebin' in list(contents.keys()):
            contents['rebin'] *= rebin
        else:
            contents['rebin'] = rebin

    elif ftype in ['pds', 'cpds']:
        x = contents['freq']
        y = contents[ftype]
        ye = contents['e' + ftype]

        # if rebin is integer, use constant rebinning. Otherwise, geometrical
        if rebin == float(int(rebin)):
            logging.info('Applying a constant rebinning')
            if do_dyn:
                old_dynspec = contents['dyn{0}'.format(ftype)]
                old_edynspec = contents['edyn{0}'.format(ftype)]

                dynspec = []
                edynspec = []
                for i_s, spec in enumerate(old_dynspec):
                    _, sp, spe = \
                        const_rebin(x, spec, rebin,
                                    old_edynspec[i_s],
                                    normalize=True)
                    dynspec.append(sp)
                    edynspec.append(spe)

                contents['dyn{0}'.format(ftype)] = np.array(dynspec)
                contents['edyn{0}'.format(ftype)] = np.array(edynspec)

            x, y, ye = \
                const_rebin(x, y, rebin, ye, normalize=True)
            contents['freq'] = x
            contents[ftype] = y
            contents['e' + ftype] = ye
            contents['rebin'] *= rebin
        else:
            logging.info('Applying a geometrical rebinning')
            if do_dyn:
                old_dynspec = contents['dyn{0}'.format(ftype)]
                old_edynspec = contents['edyn{0}'.format(ftype)]

                dynspec = []
                edynspec = []
                for i_s, spec in enumerate(old_dynspec):
                    _, _, sp, spe, _ = \
                        geom_bin(x, spec, rebin,
                                 old_edynspec[i_s],
                                 return_nbins=True)
                    dynspec.append(sp)
                    edynspec.append(spe)

                contents['dyn{0}'.format(ftype)] = np.array(dynspec)
                contents['edyn{0}'.format(ftype)] = np.array(edynspec)

            x1, x2, y, ye, nbin = \
                geom_bin(x, y, rebin, ye, return_nbins=True)
            del contents['freq']
            contents['flo'] = x1
            contents['fhi'] = x2
            contents[ftype] = y
            contents['e' + ftype] = ye
            contents['nbins'] = nbin
            contents['rebin'] *= nbin
    else:
        raise Exception('Format was not recognized:', ftype)

    outfile = filename.replace(get_file_extension(filename),
                               '_rebin%g' % rebin + MP_FILE_EXTENSION)
    logging.info('Saving %s to %s' % (ftype, outfile))
    save_data(contents, outfile, ftype)


def main(args=None):
    import argparse
    description = 'Rebin light curves and frequency spectra. '
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of light curve files", nargs='+')
    parser.add_argument("-r", "--rebin", type=float, default=1,
                        help="Rebinning to apply. Only if the quantity to" +
                        " rebin is a (C)PDS, it is possible to specify a" +
                        " non-integer rebin factor, in which case it is" +
                        " interpreted as a geometrical binning factor")

    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')

    args = parser.parse_args(args)
    files = args.files

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='MPrebin.log', level=numeric_level,
                        filemode='w')
    rebin = args.rebin
    for f in files:
        rebin_file(f, rebin)
