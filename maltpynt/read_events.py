# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Read and save event lists from FITS files."""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

from .base import mp_root, read_header_key, ref_mjd, _empty
from .base import _order_list_of_arrays
from .io import save_events
from .io import MP_FILE_EXTENSION
import numpy as np
import logging
import warnings
import os


def _get_gti_from_extension(lchdulist, accepted_gtistrings=['GTI']):
    hdunames = [h.name for h in lchdulist]
    gtiextn = [ix for ix, x in enumerate(hdunames)
               if x in accepted_gtistrings][0]
    gtiext = lchdulist[gtiextn]
    gtitable = gtiext.data

    colnames = [col.name for col in gtitable.columns.columns]
    # Default: NuSTAR: START, STOP. Otherwise, try RXTE: Start, Stop
    if 'START' in colnames:
        startstr, stopstr = 'START', 'STOP'
    else:
        startstr, stopstr = 'Start', 'Stop'

    gtistart = np.array(gtitable.field(startstr), dtype=np.longdouble)
    gtistop = np.array(gtitable.field(stopstr), dtype=np.longdouble)
    gti_list = np.array([[a, b]
                         for a, b in zip(gtistart,
                                         gtistop)],
                        dtype=np.longdouble)
    return gti_list


def _get_additional_data(lctable, additional_columns):
    additional_data = {}
    if additional_columns is not None:
        for a in additional_columns:
            try:
                additional_data[a] = np.array(lctable.field(a))
            except:  # pragma: no cover
                if a == 'PI':
                    logging.warning('Column PI not found. Trying with PHA')
                    additional_data[a] = np.array(lctable.field('PHA'))
                else:
                    raise Exception('Column' + a + 'not found')

    return additional_data


def load_gtis(fits_file, gtistring=None):
    """Load GTI from HDU EVENTS of file fits_file."""
    from astropy.io import fits as pf
    import numpy as np

    if gtistring is None:
        gtistring = 'GTI'
    logging.info("Loading GTIS from file %s" % fits_file)
    lchdulist = pf.open(fits_file, checksum=True)
    lchdulist.verify('warn')

    gtitable = lchdulist[gtistring].data
    gti_list = np.array([[a, b]
                         for a, b in zip(gtitable.field('START'),
                                         gtitable.field('STOP'))],
                        dtype=np.longdouble)
    lchdulist.close()
    return gti_list


def load_events_and_gtis(fits_file, additional_columns=None,
                         gtistring='GTI,STDGTI',
                         gti_file=None, hduname='EVENTS', column='TIME'):
    """Load event lists and GTIs from one or more files.

    Loads event list from HDU EVENTS of file fits_file, with Good Time
    intervals. Optionally, returns additional columns of data from the same
    HDU of the events.

    Parameters
    ----------
    fits_file : str
    return_limits: bool, optional
        Return the TSTART and TSTOP keyword values
    additional_columns: list of str, optional
        A list of keys corresponding to the additional columns to extract from
        the event HDU (ex.: ['PI', 'X'])

    Returns
    -------
    ev_list : array-like
    gtis: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
    additional_data: dict
        A dictionary, where each key is the one specified in additional_colums.
        The data are an array with the values of the specified column in the
        fits file.
    t_start : float
    t_stop : float
    """
    from astropy.io import fits as pf

    lchdulist = pf.open(fits_file)

    # Load data table
    try:
        lctable = lchdulist[hduname].data
    except:  # pragma: no cover
        logging.warning('HDU %s not found. Trying first extension' % hduname)
        lctable = lchdulist[1].data

    # Read event list
    ev_list = np.array(lctable.field(column), dtype=np.longdouble)

    # Read TIMEZERO keyword and apply it to events
    try:
        timezero = np.longdouble(lchdulist[1].header['TIMEZERO'])
    except:  # pragma: no cover
        logging.warning("No TIMEZERO in file")
        timezero = np.longdouble(0.)

    ev_list += timezero

    # Read TSTART, TSTOP from header
    try:
        t_start = np.longdouble(lchdulist[1].header['TSTART'])
        t_stop = np.longdouble(lchdulist[1].header['TSTOP'])
    except:  # pragma: no cover
        logging.warning("Tstart and Tstop error. using defaults")
        t_start = ev_list[0]
        t_stop = ev_list[-1]

    # Read and handle GTI extension
    accepted_gtistrings = gtistring.split(',')

    if gti_file is None:
        # Select first GTI with accepted name
        try:
            gti_list = \
                _get_gti_from_extension(
                    lchdulist, accepted_gtistrings=accepted_gtistrings)
        except:  # pragma: no cover
            warnings.warn("No extensions found with a valid name. "
                          "Please check the `accepted_gtistrings` values.")
            gti_list = np.array([[t_start, t_stop]],
                                dtype=np.longdouble)
    else:
        gti_list = load_gtis(gti_file, gtistring)

    additional_data = _get_additional_data(lctable, additional_columns)

    lchdulist.close()

    # Sort event list
    order = np.argsort(ev_list)
    ev_list = ev_list[order]

    additional_data = _order_list_of_arrays(additional_data, order)

    returns = _empty()
    returns.ev_list = ev_list
    returns.gti_list = gti_list
    returns.additional_data = additional_data
    returns.t_start = t_start
    returns.t_stop = t_stop

    return returns


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
