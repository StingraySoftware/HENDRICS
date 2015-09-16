# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to calculate lags."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .fspec import read_fspec
from .base import mp_root, _assign_value_if_none
from .io import MP_FILE_EXTENSION, save_data, load_data
import numpy as np
import logging
import os


def calc_lags(freqs, cpds, pds1, pds2, n_chunks, rebin):
    """Calculate time lags.

    Parameters
    ----------
    freqs : array-like
        The frequency array
    cpds : array-like
        The cross power spectrum
    pds1 : array-like
        The PDS of the first channel
    pds2 : array-like
        The PDS of the second channel
    n_chunks : int or array-like
        The number of PDSs averaged
    rebin : int or array-like
        The number of bins averaged to obtain each bin

    Returns
    -------
    lags : array-like
        The lags spectrum at frequencies corresponding to freqs
    lagse : array-like
        The errors on `lags`
    """
    lags = np.angle(cpds) / (2 * np.pi * freqs)
    sigcpd = np.absolute(cpds)

    rawcof = (sigcpd) ** 2 / ((pds1) * (pds2))

    dum = (1. - rawcof) / (2. * rawcof)

    lagse = np.sqrt(dum / n_chunks / rebin) / (2 * np.pi * freqs)

    bad = np.logical_or(lagse != lagse, lags != lags)

    if np.any(bad):
        logging.error('Bad element(s) in lag or lag error array:')
        fbad = freqs[bad]
        lbad = lags[bad]
        lebad = lagse[bad]
        cbad = cpds[bad]
        p1bad = pds1[bad]
        p2bad = pds2[bad]

        for i, f in enumerate(fbad):
            logging.error('--------------------------------------------------')
            logging.error('Freq (Hz), Lag, Lag_err, CPDS (x + jy), PDS1, PDS2')
            logging.error('--------------------------------------------------')
            logging.error(" ".join([repr(fbad[i]),
                                    repr(lbad[i]),
                                    repr(lebad[i]),
                                    repr(cbad[i]),
                                    repr(p1bad[i]),
                                    repr(p2bad[i])]
                                   )
                          )
        lags[bad] = 0
        lagse[bad] = 0

    return lags, lagse


def lags_from_spectra(cpdsfile, pds1file, pds2file, outroot='lag',
                      noclobber=False):
    """Calculate time lags.

    Parameters
    ----------
    cpdsfile : str
        The MP-format file containing the CPDS
    pds1file : str
        The MP-format file containing the first PDS used for the CPDS
    pds1file : str
        The MP-format file containing the second PDS

    Returns
    -------
    freq : array-like
        Central frequencies of spectral bins
    df : array-like
        Width of each spectral bin
    lags : array-like
        Time lags
    elags : array-like
        Error on the time lags

    Other Parameters
    ----------------
    outroot : str
        Root of the output filename
    noclobber : bool
        If True, do not overwrite existing files
    """
    warn = ("----------------- lags_from_spectra -----------------\n\n"
            "This program is still under testing and no assurance of\n"
            "validity is provided. If you'd like to help, please test\n"
            "this first on known data!\n\n"
            "--------------------------------------------------------")

    logging.warning(warn)

    outname = outroot + "_lag" + MP_FILE_EXTENSION
    if noclobber and os.path.exists(outname):
        print('File exists, and noclobber option used. Skipping')
        contents = load_data(outname)
        return contents['freq'], contents['df'], \
            contents['lags'], contents['elags']

    ftype,  cfreq, cpds, ecpds, nchunks, rebin, ccontents = \
        read_fspec(cpdsfile)
    ftype, p1freq, pds1, epds1, nchunks, rebin, p1contents = \
        read_fspec(pds1file)
    ftype, p2freq, pds2, epds2, nchunks, rebin, p2contents = \
        read_fspec(pds2file)
    ctime = ccontents['time']
    fftlen = ccontents['fftlen']
    instrs = ccontents['Instrs']
    ctrate = ccontents['ctrate']
    tctrate = ccontents['total_ctrate']

    if 'reb' in ftype:
        flo, fhi = cfreq
        freq = (flo + fhi) / 2
        df = (fhi - flo)
    else:
        freq = cfreq
        df = np.zeros(len(freq)) + (freq[1] - freq[0])

    logging.debug(repr(cpds))
    logging.debug(repr(pds1))
    logging.debug(repr(pds1))
    logging.debug(len(cpds))
    logging.debug(len(pds1))
    logging.debug(len(pds1))

    assert len(cpds) == len(pds1), 'Files are not compatible'
    assert len(cpds) == len(pds2), 'Files are not compatible'

    lags, elags = calc_lags(freq, cpds, pds1, pds2, nchunks, rebin)

    outdata = {'time': ctime, 'lag': lags, 'elag': elags, 'ncpds': nchunks,
               'fftlen': fftlen, 'Instrs': instrs,
               'freq': freq, 'df': df, 'rebin': rebin,
               'ctrate': ctrate, 'total_ctrate': tctrate}

    logging.info('Saving lags to %s' % outname)
    save_data(outdata, outname)

    return freq, df, lags, elags


def main(args=None):
    import argparse

    description = ('Calculate time lags from the cross power spectrum and '
                   'the power spectra of the two channels')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="Three files: the cross spectrum" +
                        " and the two power spectra", nargs='+')
    parser.add_argument("-o", "--outroot", type=str, default=None,
                        help='Root of output file names')
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG;"
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--noclobber", help="Do not overwrite existing files",
                        default=False, action='store_true')
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='MPlags.log', level=numeric_level,
                        filemode='w')

    if len(args.files) != 3:
        raise Exception('Invalid number of arguments')
    cfile, p1file, p2file = args.files

    outroot = _assign_value_if_none(args.outroot, mp_root(cfile) + '_lag')

    f, df, l, le = lags_from_spectra(cfile, p1file, p2file, outroot=outroot,
                                     noclobber=args.noclobber)
