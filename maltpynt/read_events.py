# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Read and save event lists from FITS files."""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .base import mp_root, read_header_key, ref_mjd
from .io import save_events, load_events_and_gtis
from .io import MP_FILE_EXTENSION
import numpy as np
import logging
import warnings
import os


def treat_event_file(filename, noclobber=False, gti_split=False,
                     min_length=4):
    """Read data from an event file, with no external GTI information.

    Parameters
    ----------
    filename : str

    Other Parameters
    ----------------
    noclobber: bool
        if a file is present, do not overwrite it
    gti_split: bool
        split the file in multiple chunks, containing one GTI each
    min_length: float
        minimum length of GTIs accepted (only if gti_split is True)
    """
    logging.info('Opening %s' % filename)
    outfile = mp_root(filename) + '_ev' + MP_FILE_EXTENSION
    if noclobber and os.path.exists(outfile) and (not gti_split):
        warnings.warn(
            '{0} exists, and noclobber option used. Skipping'.format(outfile))
        return

    instr = read_header_key(filename, 'INSTRUME')
    additional_columns = ['PI']
    if instr == 'PCA':
        additional_columns.append('PCUID')

    mjdref = ref_mjd(filename)
    data = load_events_and_gtis(filename,
                                additional_columns=additional_columns)

    events = data.ev_list
    gtis = data.gti_list
    additional = data.additional_data
    tstart = data.t_start
    tstop = data.t_stop

    pis = additional['PI']
    out = {'time': events,
           'GTI': gtis,
           'PI': pis,
           'MJDref': mjdref,
           'Tstart': tstart,
           'Tstop': tstop,
           'Instr': instr
           }

    if instr == "PCA":
        out['PCU'] = np.array(additional['PCUID'], dtype=np.byte)

    if gti_split:
        for ig, g in enumerate(gtis):
            length = g[1] - g[0]
            if length < min_length:
                print("GTI shorter than {0} s; skipping".format(min_length))
                continue

            outfile_local = \
                '{0}_{1}'.format(outfile.replace(MP_FILE_EXTENSION,
                                                 ''), ig) + \
                MP_FILE_EXTENSION
            if noclobber and os.path.exists(outfile_local):
                warnings.warn('{0} exists, '.format(outfile_local) +
                              'and noclobber option used. Skipping')
                return
            out_local = out.copy()
            good = np.logical_and(events >= g[0], events < g[1])
            if not np.any(good):
                print("This GTI has no valid events; skipping")
                continue
            out_local['time'] = events[good]
            out_local['Tstart'] = g[0]
            out_local['Tstop'] = g[1]
            out_local['PI'] = pis[good]
            out_local['GTI'] = np.array([g], dtype=np.longdouble)
            save_events(out_local, outfile_local)
        pass
    else:
        save_events(out, outfile)


def _wrap_fun(arglist):
    f, kwargs = arglist
    return treat_event_file(f, **kwargs)


def main(args=None):
    """Main function called by the `MPreadevents` command line script."""
    import argparse
    from multiprocessing import Pool

    description = ('Read a cleaned event files and saves the relevant '
                   'information in a standard format')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--nproc",
                        help=("Number of processors to use"),
                        default=1,
                        type=int)
    parser.add_argument("--noclobber",
                        help=("Do not overwrite existing event files"),
                        default=False, action='store_true')
    parser.add_argument("-g", "--gti-split",
                        help="Split event list by GTI",
                        default=False,
                        action="store_true")
    parser.add_argument("--min-length", type=int,
                        help="Minimum length of GTIs to consider",
                        default=0)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')

    args = parser.parse_args(args)
    files = args.files

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='MPreadevents.log', level=numeric_level,
                        filemode='w')

    argdict = {"noclobber": args.noclobber, "gti_split": args.gti_split,
               "min_length": args.min_length}
    arglist = [[f, argdict] for f in files]

    if os.name == 'nt' or args.nproc == 1:
        [_wrap_fun(a) for a in arglist]
    else:
        pool = Pool(processes=args.nproc)
        for i in pool.imap_unordered(_wrap_fun, arglist):
            pass
        pool.close()
