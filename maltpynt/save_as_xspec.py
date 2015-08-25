# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to save data in a Xspec-readable format."""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .io import get_file_type
import numpy as np
from .io import get_file_extension
import subprocess as sp
import logging


def save_as_xspec(fname, direct_save=False):
    """Save frequency spectra in a format readable to FTOOLS and Xspec.

    Parameters
    ----------
    fname : str
        Input MaLTPyNT frequency spectrum file name
    direct_save : bool
        If True, also call `flx2xsp` to produce the output .pha and .rsp files.
        If False (default), flx2xsp has to be called from the user

    Notes
    -----
    Uses method described here:
    https://asd.gsfc.nasa.gov/XSPECwiki/fitting_timing_power_spectra_in_XSPEC
    """
    ftype, contents = get_file_type(fname)

    outroot = fname.replace(get_file_extension(fname), '')
    outname = outroot + '_xsp.dat'

    if 'freq' in list(contents.keys()):
        freq = contents['freq']
        pds = contents[ftype]
        epds = contents['e' + ftype]
        df = freq[1] - freq[0]

        np.savetxt(outname, np.transpose([freq - df / 2,
                                          freq + df / 2,
                                          pds.real * df,
                                          epds * df]))
    elif 'flo' in list(contents.keys()):
        ftype = ftype.replace('reb', '')
        flo = contents['flo']
        fhi = contents['fhi']
        pds = contents[ftype]
        epds = contents['e' + ftype]
        df = fhi - flo
        np.savetxt(outname, np.transpose([flo, fhi,
                                          pds.real * df,
                                          epds * df]))
    else:
        raise Exception('File type not recognized')

    if direct_save:
        sp.check_call('flx2xsp {0} {1}.pha {1}.rsp'.format(
            outname, outroot).split())


def main(args=None):
    import argparse
    description = ('Save a frequency spectrum in a qdp file that can be '
                   'read by flx2xsp and produce a XSpec-compatible spectrum'
                   'file')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')
    parser.add_argument("--flx2xsp", help="Also call flx2xsp at the end",
                        default=False, action='store_true')

    args = parser.parse_args(args)
    files = args.files
    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='MP2xpec.log', level=numeric_level,
                        filemode='w')

    for f in files:
        save_as_xspec(f, direct_save=args.flx2xsp)
