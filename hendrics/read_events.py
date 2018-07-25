# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Read and save event lists from FITS files."""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from stingray.utils import assign_value_if_none
from stingray.events import EventList
from .base import hen_root, read_header_key
from .io import save_events, load_events_and_gtis
from .io import HEN_FILE_EXTENSION
import numpy as np
import logging
import warnings
import os


def treat_event_file(filename, noclobber=False, gti_split=False,
                     min_length=4, gtistring=None):
    """Read data from an event file, with no external GTI information.

    Parameters
    ----------
    filename : str

    Other Parameters
    ----------------
    noclobber: bool
        if a file is present, do not overwrite it
    gtistring: str
        comma-separated set of GTI strings to consider
    gti_split: bool
        split the file in multiple chunks, containing one GTI each
    min_length: float
        minimum length of GTIs accepted (only if gti_split is True)
    """
    gtistring = assign_value_if_none(gtistring, 'GTI,STDGTI')
    logging.info('Opening %s' % filename)

    instr = read_header_key(filename, 'INSTRUME')
    mission = read_header_key(filename, 'TELESCOP')

    data = load_events_and_gtis(filename,
                                gtistring=gtistring)

    events = data.ev_list
    gtis = events.gti
    detector_id = data.detector_id

    if detector_id is not None:
        detectors = np.array(list(set(detector_id)))
    else:
        detectors = [None]
    outfile_root = \
        hen_root(filename) + '_' + mission.lower() + '_' + instr.lower()

    for d in detectors:
        if d is not None:
            good_det = d == data.detector_id
            outroot_local = \
                '{0}_det{1:02d}'.format(outfile_root, d)

        else:
            good_det = np.ones_like(events.time, dtype=bool)
            outroot_local = outfile_root

        outfile = outroot_local + '_ev' + HEN_FILE_EXTENSION
        if noclobber and os.path.exists(outfile) and (not gti_split):
            warnings.warn(
                '{0} exists and using noclobber. Skipping'.format(outfile))
            return

        if gti_split:
            for ig, g in enumerate(gtis):
                length = g[1] - g[0]
                if length < min_length:
                    print("GTI shorter than {} s; skipping".format(min_length))
                    continue

                outfile_local = \
                    '{0}_gti{1}_ev'.format(outroot_local,
                                           ig) + HEN_FILE_EXTENSION

                if noclobber and os.path.exists(outfile_local):
                    warnings.warn('{0} exists, '.format(outfile_local) +
                                  'and noclobber option used. Skipping')
                    return
                good = np.logical_and(events.time >= g[0],
                                      events.time < g[1])
                all_good = good_det & good
                if len(events.time[all_good]) < 1:
                    continue
                events_filt = EventList(events.time[all_good],
                                        pi=events.pi[all_good],
                                        gti=np.array([g], dtype=np.longdouble),
                                        mjdref=events.mjdref)
                events_filt.instr = events.instr
                events_filt.header = events.header
                save_events(events_filt, outfile_local)
            pass
        else:
            events_filt = EventList(events.time[good_det],
                                    pi=events.pi[good_det],
                                    gti=events.gti, mjdref=events.mjdref)
            events_filt.instr = events.instr
            events_filt.header = events.header

            save_events(events_filt, outfile)


def _wrap_fun(arglist):
    f, kwargs = arglist
    return treat_event_file(f, **kwargs)


def main(args=None):
    """Main function called by the `HENreadevents` command line script."""
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
    parser.add_argument("--gti-string", type=str,
                        help="GTI string",
                        default=None)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')

    args = parser.parse_args(args)
    files = args.files

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='HENreadevents.log', level=numeric_level,
                        filemode='w')

    argdict = {"noclobber": args.noclobber, "gti_split": args.gti_split,
               "min_length": args.min_length, "gtistring": args.gti_string}

    arglist = [[f, argdict] for f in files]

    if os.name == 'nt' or args.nproc == 1:
        [_wrap_fun(a) for a in arglist]
    else:
        pool = Pool(processes=args.nproc)
        for i in pool.imap_unordered(_wrap_fun, arglist):
            pass
        pool.close()
