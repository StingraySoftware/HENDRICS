# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to rebin light curves and frequency spectra."""

import numpy as np
from .io import get_file_type
from .io import save_lcurve, save_pds
from .io import HEN_FILE_EXTENSION, get_file_extension
from .base import _empty, _assign_value_if_none
from astropy import log


def rebin_file(filename, rebin):
    """Rebin the contents of a file, be it a light curve or a spectrum."""
    ftype, contents = get_file_type(filename)

    if ftype not in ['lc', 'pds', 'cpds']:
        raise ValueError('This format does not support rebin (yet):', ftype)

    if rebin == np.int(rebin):
        contents = contents.rebin(f=rebin)
    else:
        contents = contents.rebin_log(f=rebin)

    if ftype == 'lc':
        func = save_lcurve
    elif ftype in ['pds', 'cpds']:
        func = save_pds

    outfile = filename.replace(get_file_extension(filename),
                               '_rebin%g' % rebin + HEN_FILE_EXTENSION)
    log.info('Saving %s to %s' % (ftype, outfile))
    func(contents, outfile)


def main(args=None):
    """Main function called by the `HENrebin` command line script."""
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

    log.setLevel(args.loglevel)


    with log.log_to_file('HENrebin.log'):
        rebin = args.rebin
        for f in files:
            rebin_file(f, rebin)
