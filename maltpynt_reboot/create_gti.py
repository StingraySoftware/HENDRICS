# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to create and apply GTIs."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .io import MP_FILE_EXTENSION, save_data, load_data, get_file_type
from .base import create_gti_from_condition, mp_root, create_gti_mask
from .base import cross_gtis, _assign_value_if_none
import logging
import numpy as np
import sys


def filter_gti_by_length(gti, minimum_length):
    """Filter a list of GTIs: keep those longer than `minimum_length`."""
    if minimum_length == 0 or minimum_length is None:
        return gti

    newgtis = []
    for g in gti:
        length = g[1] - g[0]
        if length >= minimum_length:
            newgtis.append(g)

    return np.array(newgtis)


def create_gti(fname, filter_expr, safe_interval=[0, 0], outfile=None,
               minimum_length=0):
    """Create a GTI list by using boolean operations on file data.

    Parameters
    ----------
    fname : str
        File name. The file must be in MaLTPyNT format.
    filter_expr : str
        A boolean condition on one or more of the arrays contained in the data.
        E.g. '(lc > 10) & (lc < 20)'

    Returns
    -------
    gtis : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        The newly created GTIs

    Other parameters
    ----------------
    safe_interval : float or [float, float]
        A safe interval to exclude at both ends (if single float) or the start
        and the end (if pair of values) of GTIs.
    outfile : str
        The output file name. If None, use a default root + '_gti' combination
    """
    # Necessary as nc variables are sometimes defined as array
    from numpy import array  # NOQA

    ftype, data = get_file_type(fname)

    instr = data['Instr']
    if ftype == 'lc' and instr == 'PCA':
        logging.warning('RXTE/PCA data; normalizing lc per no. PCUs')
        # If RXTE, plot per PCU count rate
        data['lc'] /= data['nPCUs']
    mjdref = data['MJDref']
    # Map all entries of data to local variables
    locals().update(data)

    good = eval(filter_expr)

    gtis = create_gti_from_condition(locals()['time'], good,
                                     safe_interval=safe_interval)

    gtis = filter_gti_by_length(gtis, minimum_length)

    outfile = _assign_value_if_none(
        outfile, mp_root(fname) + '_gti' + MP_FILE_EXTENSION)
    save_data({'GTI': gtis, 'MJDref': mjdref}, outfile)

    return gtis


def apply_gti(fname, gti, outname=None,
              minimum_length=0):
    """Apply a GTI list to the data contained in a file.

    File MUST have a GTI extension already, and an extension called `time`.
    """
    ftype, data = get_file_type(fname)

    try:
        datagti = data['GTI']
        newgtis = cross_gtis([gti, datagti])
    except:  # pragma: no cover
        logging.warning('Data have no GTI extension')
        newgtis = gti

    newgtis = filter_gti_by_length(newgtis, minimum_length)

    data['GTI'] = newgtis
    good = create_gti_mask(data['time'], newgtis)
    data['time'] = data['time'][good]
    if ftype == 'lc':
        data['lc'] = data['lc'][good]
    elif ftype == 'events':
        data['PI'] = data['PI'][good]
        if data['Instr'] == 'PCA':  # pragma: no cover
            data['PCU'] = data['PCU'][good]

    outname = _assign_value_if_none(
        outname,
        fname.replace(MP_FILE_EXTENSION, '') + '_gtifilt' + MP_FILE_EXTENSION)
    save_data(data, outname)

    return newgtis


def main(args=None):
    """Main function called by the `MPcreategti` command line script."""
    import argparse

    description = ('Create GTI files from a filter expression, or applies '
                   'previously created GTIs to a file')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs='+')

    parser.add_argument("-f", "--filter", type=str, default=None,
                        help="Filter expression, that has to be a valid " +
                        "Python boolean operation on a data variable " +
                        "contained in the files")

    parser.add_argument("-c", "--create-only",
                        default=False, action="store_true",
                        help="If specified, creates GTIs withouth applying" +
                        "them to files (Default: False)")

    parser.add_argument("--overwrite",
                        default=False, action="store_true",
                        help="Overwrite original file (Default: False)")

    parser.add_argument("-a", "--apply-gti", type=str, default=None,
                        help="Apply a GTI from this file to input files")

    parser.add_argument("-l", "--minimum-length", type=float, default=0,
                        help=("Minimum length of GTIs (below this length, they"
                              " will be discarded)"))

    parser.add_argument("--safe-interval", nargs=2, type=float,
                        default=[0, 0],
                        help="Interval at start and stop of GTIs used" +
                        " for filtering")

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
    logging.basicConfig(filename='MPcreategti.log', level=numeric_level,
                        filemode='w')

    filter_expr = args.filter
    if filter_expr is None and args.apply_gti is None:
        sys.exit("Please specify filter expression (-f option) or input "
                 "GTI file (-a option)")

    for fname in files:
        if args.apply_gti is not None:
            data = load_data(args.apply_gti)
            gtis = data['GTI']
        else:
            gtis = create_gti(fname, filter_expr,
                              safe_interval=args.safe_interval,
                              minimum_length=args.minimum_length)
        if args.create_only:
            continue
        if args.overwrite:
            outname = fname
        else:
            # Use default
            outname = None
        apply_gti(fname, gtis, outname=outname,
                  minimum_length=args.minimum_length)
