# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Light curve-related functions."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
import os
import logging
import warnings
import copy

from stingray.lightcurve import Lightcurve
from stingray.utils import assign_value_if_none
from stingray.gti import create_gti_mask, cross_gtis, contiguous_regions
from .base import _look_for_array_in_array, hen_root, mkdir_p
from .io import load_events, load_data, save_data, save_lcurve, load_lcurve
from .io import HEN_FILE_EXTENSION, high_precision_keyword_read, get_file_type


def join_lightcurves(lcfilelist, outfile='out_lc' + HEN_FILE_EXTENSION):
    """Join light curves from different files.

    Light curves from different instruments are put in different channels.

    Parameters
    ----------
    lcfilelist : list of str
        List of input file names
    outfile :
        Output light curve
    See Also
    --------
        scrunch_lightcurves : Create a single light curve from input light
                                 curves.

    """
    lcdatas = []

    for lfc in lcfilelist:
        logging.info("Loading file %s..." % lfc)
        lcdata = load_lcurve(lfc)
        logging.info("Done.")
        lcdatas.append(lcdata)
        del lcdata

    # --------------- Check consistency of data --------------
    lcdts = [lcdata.dt for lcdata in lcdatas]
    # Find unique elements. If multiple bin times are used, throw an exception
    lcdts = list(set(lcdts))
    assert len(lcdts) == 1, 'Light curves must have same dt for joining'

    instrs = [lcdata.instr for lcdata in lcdatas if hasattr(lcdata, 'instr')]

    # Find unique elements. A lightcurve will be produced for each instrument
    instrs = list(set(instrs))
    if instrs == []:
        instrs = ['unknown']

    outlcs = {}
    for instr in instrs:
        outlcs[instr] = None
    # -------------------------------------------------------

    for lcdata in lcdatas:
        instr = assign_value_if_none(lcdata.instr, 'unknown')
        if outlcs[instr] is None:
            outlcs[instr] = lcdata
        else:
            outlcs[instr] = outlcs[instr].join(lcdata)

    if outfile is not None:
        for instr in instrs:
            if len(instrs) == 1:
                tag = ""
            else:
                tag = instr
            logging.info('Saving joined light curve to %s' % outfile)

            dname, fname = os.path.split(outfile)
            save_lcurve(outlcs[instr], os.path.join(dname, tag + fname))

    return outlcs


def scrunch_lightcurves(lcfilelist, outfile='out_scrlc'+HEN_FILE_EXTENSION,
                        save_joint=False):
    """Create a single light curve from input light curves.

    Light curves are appended when they cover different times, and summed when
    they fall in the same time range. This is done regardless of the channel
    or the instrument.

    Parameters
    ----------
    lcfilelist : list of str
        The list of light curve files to scrunch

    Returns
    -------
    time : array-like
        The time array
    lc :
        The new light curve
    gti : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good Time Intervals

    Other Parameters
    ----------------
    outfile : str
        The output file name
    save_joint : bool
        If True, save the per-channel joint light curves

    See Also
    --------
        join_lightcurves : Join light curves from different files
    """
    if save_joint:
        lcdata = join_lightcurves(lcfilelist)
    else:
        lcdata = join_lightcurves(lcfilelist, outfile=None)

    instrs = list(lcdata.keys())
    gti_lists = [lcdata[inst].gti for inst in instrs]
    gti = cross_gtis(gti_lists)
    # Determine limits

    lc0 = lcdata[instrs[0]]

    for inst in instrs[1:]:
        lc0 = lc0 + lcdata[inst]

    lc0.instr = ",".join(instrs)

    logging.info('Saving scrunched light curve to %s' % outfile)
    save_lcurve(lc0, outfile)

    return lc0


def filter_lc_gtis(lc, safe_interval=None, delete=False,
                   min_length=0, return_borders=False):
    """Filter a light curve for GTIs.

    Parameters
    ----------
    lc : :class:`Lightcurve` object
        The input light curve

    Returns
    -------
    newlc : :class:`Lightcurve` object
        The output light curve
    borders : [[i0_0, i0_1], [i1_0, i1_1], ...], optional
        The indexes of the light curve corresponding to the borders of the
        GTIs. Returned if return_borders is set to True

    Other Parameters
    ----------------
    safe_interval : float or [float, float]
        Seconds to filter out at the start and end of each GTI. If single
        float, these safe windows are equal, otherwise the two numbers refer
        to the start and end of the GTI respectively
    delete : bool
        If delete is True, the intervals outside of GTIs are filtered out from
        the light curve. Otherwise, they are set to zero.
    min_length : float
        Minimum length of GTI. GTIs below this length will be removed.
    return_borders : bool
        If True, return also the indexes of the light curve corresponding to
        the borders of the GTIs
    """
    mask, newgtis = create_gti_mask(lc.time, lc.gti,
                                    return_new_gtis=True,
                                    safe_interval=safe_interval,
                                    min_length=min_length)

    nomask = np.logical_not(mask)

    newlc = copy.copy(lc)
    newlc.counts[nomask] = 0
    newlc.gti = newgtis

    if return_borders:
        mask = create_gti_mask(lc.time, newgtis)
        borders = contiguous_regions(mask)
        return newlc, borders
    else:
        return newlc


def lcurve_from_events(f, safe_interval=0,
                       pi_interval=None,
                       e_interval=None,
                       min_length=0,
                       gti_split=False,
                       ignore_gtis=False,
                       bintime=1.,
                       outdir=None,
                       outfile=None,
                       noclobber=False):
    """Bin an event list in a light curve.

    Parameters
    ----------
    f : str
        Input event file name
    bintime : float
        The bin time of the output light curve

    Returns
    -------
    outfiles : list
        List of output light curves

    Other Parameters
    ----------------
    safe_interval : float or [float, float]
        Seconds to filter out at the start and end of each GTI. If single
        float, these safe windows are equal, otherwise the two numbers refer
        to the start and end of the GTI respectively
    pi_interval : [int, int]
        PI channel interval to select. Default None, meaning that all PI
        channels are used
    e_interval : [float, float]
        Energy interval to select (only works if event list is calibrated with
        `calibrate`). Default None
    min_length : float
        GTIs below this length will be filtered out
    gti_split : bool
        If True, create one light curve for each good time interval
    ignore_gtis : bool
        Ignore good time intervals, and get a single light curve that includes
        possible gaps
    outdir : str
        Output directory
    outfile : str
        Output file
    noclobber : bool
        If True, do not overwrite existing files

    """
    logging.info("Loading file %s..." % f)
    evdata = load_events(f)
    logging.info("Done.")

    if bintime < 0:
        bintime = 2 ** (bintime)
    bintime = np.longdouble(bintime)

    tag = ''

    gtis = evdata.gti
    tstart = np.min(gtis)
    tstop = np.max(gtis)
    events = evdata.time
    if hasattr(evdata, 'instr') and evdata.instr is not None:
        instr = evdata.instr
    else:
        instr = "unknown"

    if ignore_gtis:
        gtis = np.array([[tstart, tstop]])
        evdata.gtis = gtis

    total_lc = evdata.to_lc(100)
    total_lc.instr = instr

    # Then, apply filters
    if pi_interval is not None and np.all(np.array(pi_interval) > 0):
        pis = evdata.pi
        good = np.logical_and(pis > pi_interval[0],
                              pis <= pi_interval[1])
        events = events[good]
        tag = '_PI%g-%g' % (pi_interval[0], pi_interval[1])
    elif e_interval is not None and np.all(np.array(e_interval) > 0):
        if not hasattr(evdata, 'energy') or evdata.energy is None:
            raise \
                ValueError("No energy information is present in the file." +
                           " Did you run HENcalibrate?")
        es = evdata.energy
        good = np.logical_and(es > e_interval[0],
                              es <= e_interval[1])
        events = events[good]
        tag = '_E%g-%g' % (e_interval[0], e_interval[1])
    else:
        pass

    if tag != "":
        save_lcurve(total_lc, hen_root(f) + '_std_lc' + HEN_FILE_EXTENSION)

    # Assign default value if None
    outfile = assign_value_if_none(outfile,  hen_root(f) + tag + '_lc')

    # Take out extension from name, if present, then give extension. This
    # avoids multiple extensions
    outfile = outfile.replace(HEN_FILE_EXTENSION, '') + HEN_FILE_EXTENSION
    outdir = assign_value_if_none(
        outdir, os.path.dirname(os.path.abspath(f)))

    _, outfile = os.path.split(outfile)
    mkdir_p(outdir)
    outfile = os.path.join(outdir, outfile)

    if noclobber and os.path.exists(outfile):
        warnings.warn('File exists, and noclobber option used. Skipping')
        return [outfile]

    lc = Lightcurve.make_lightcurve(events, bintime, tstart=tstart,
                                    tseg=tstop-tstart, mjdref=evdata.mjdref,
                                    gti=gtis)

    lc.instr = instr

    lc = filter_lc_gtis(lc, safe_interval=safe_interval,
                        delete=False,
                        min_length=min_length)

    if len(lc.gti) == 0:
        warnings.warn(
            "No GTIs above min_length ({0}s) found.".format(min_length))
        return

    lc.header = None
    if hasattr(evdata, 'header'):
        lc.header = evdata.header

    if gti_split:
        lcs = lc.split_by_gti()
        outfiles = []

        for ib, l0 in enumerate(lcs):
            local_tag = tag + '_gti{:03d}'.format(ib)
            outf = hen_root(outfile) + local_tag + '_lc' + HEN_FILE_EXTENSION
            if noclobber and os.path.exists(outf):
                warnings.warn(
                    'File exists, and noclobber option used. Skipping')
                outfiles.append(outf)
            l0.instr = lc.instr
            l0.header = lc.header

            save_lcurve(l0, outf)
            outfiles.append(outf)
    else:
        logging.info('Saving light curve to %s' % outfile)
        save_lcurve(lc, outfile)
        outfiles = [outfile]

    # For consistency in return value
    return outfiles


def lcurve_from_fits(fits_file, gtistring='GTI',
                     timecolumn='TIME', ratecolumn=None, ratehdu=1,
                     fracexp_limit=0.9, outfile=None,
                     noclobber=False, outdir=None):
    """
    Load a lightcurve from a fits file and save it in HENDRICS format.

    .. note ::
        FITS light curve handling is still under testing.
        Absolute times might be incorrect depending on the light curve format.

    Parameters
    ----------
    fits_file : str
        File name of the input light curve in FITS format

    Returns
    -------
    outfile : [str]
        Returned as a list with a single element for consistency with
        `lcurve_from_events`

    Other Parameters
    ----------------
    gtistring : str
        Name of the GTI extension in the FITS file
    timecolumn : str
        Name of the column containing times in the FITS file
    ratecolumn : str
        Name of the column containing rates in the FITS file
    ratehdu : str or int
        Name or index of the FITS extension containing the light curve
    fracexp_limit : float
        Minimum exposure fraction allowed
    outfile : str
        Output file name
    noclobber : bool
        If True, do not overwrite existing files
    """
    logging.warning(
        """WARNING! FITS light curve handling is still under testing.
        Absolute times might be incorrect.""")
    # TODO:
    # treat consistently TDB, UTC, TAI, etc. This requires some documentation
    # reading. For now, we assume TDB
    from astropy.io import fits as pf
    from astropy.time import Time
    import numpy as np
    from stingray.gti import create_gti_from_condition

    outfile = assign_value_if_none(outfile, hen_root(fits_file) + '_lc')
    outfile = outfile.replace(HEN_FILE_EXTENSION, '') + HEN_FILE_EXTENSION
    outdir = assign_value_if_none(
        outdir, os.path.dirname(os.path.abspath(fits_file)))

    _, outfile = os.path.split(outfile)
    mkdir_p(outdir)
    outfile = os.path.join(outdir, outfile)

    if noclobber and os.path.exists(outfile):
        warnings.warn('File exists, and noclobber option used. Skipping')
        return [outfile]

    lchdulist = pf.open(fits_file)
    lctable = lchdulist[ratehdu].data

    # Units of header keywords
    tunit = lchdulist[ratehdu].header['TIMEUNIT']

    try:
        mjdref = high_precision_keyword_read(lchdulist[ratehdu].header,
                                             'MJDREF')
        mjdref = Time(mjdref, scale='tdb', format='mjd')
    except:
        mjdref = None

    try:
        instr = lchdulist[ratehdu].header['INSTRUME']
    except:
        instr = 'EXTERN'

    # ----------------------------------------------------------------
    # Trying to comply with all different formats of fits light curves.
    # It's a madness...
    try:
        tstart = high_precision_keyword_read(lchdulist[ratehdu].header,
                                             'TSTART')
        tstop = high_precision_keyword_read(lchdulist[ratehdu].header,
                                            'TSTOP')
    except:
        raise(Exception('TSTART and TSTOP need to be specified'))

    # For nulccorr lcs this whould work

    timezero = high_precision_keyword_read(lchdulist[ratehdu].header,
                                           'TIMEZERO')
    # Sometimes timezero is "from tstart", sometimes it's an absolute time.
    # This tries to detect which case is this, and always consider it
    # referred to tstart
    timezero = assign_value_if_none(timezero, 0)

    # for lcurve light curves this should instead work
    if tunit == 'd':
        # TODO:
        # Check this. For now, I assume TD (JD - 2440000.5).
        # This is likely wrong
        timezero = Time(2440000.5 + timezero, scale='tdb', format='jd')
        tstart = Time(2440000.5 + tstart, scale='tdb', format='jd')
        tstop = Time(2440000.5 + tstop, scale='tdb', format='jd')
        # if None, use NuSTAR defaulf MJDREF
        mjdref = assign_value_if_none(
            mjdref, Time(np.longdouble('55197.00076601852'), scale='tdb',
                         format='mjd'))

        timezero = (timezero - mjdref).to('s').value
        tstart = (tstart - mjdref).to('s').value
        tstop = (tstop - mjdref).to('s').value

    if timezero > tstart:
        timezero -= tstart

    time = np.array(lctable.field(timecolumn), dtype=np.longdouble)
    if time[-1] < tstart:
        time += timezero + tstart
    else:
        time += timezero

    try:
        dt = high_precision_keyword_read(lchdulist[ratehdu].header,
                                         'TIMEDEL')
        if tunit == 'd':
            dt *= 86400
    except:
        warnings.warn('Assuming that TIMEDEL is the difference between the'
                      ' first two times of the light curve')
        dt = time[1] - time[0]

    # ----------------------------------------------------------------
    ratecolumn = assign_value_if_none(
        ratecolumn,
        _look_for_array_in_array(['RATE', 'RATE1', 'COUNTS'], lctable.names))

    rate = np.array(lctable.field(ratecolumn), dtype=np.float)

    try:
        rate_e = np.array(lctable.field('ERROR'), dtype=np.longdouble)
    except:
        rate_e = np.zeros_like(rate)

    if 'RATE' in ratecolumn:
        rate *= dt
        rate_e *= dt

    try:
        fracexp = np.array(lctable.field('FRACEXP'), dtype=np.longdouble)
    except:
        fracexp = np.ones_like(rate)

    good_intervals = (rate == rate) * (fracexp >= fracexp_limit) * \
        (fracexp <= 1)

    rate[good_intervals] /= fracexp[good_intervals]
    rate_e[good_intervals] /= fracexp[good_intervals]

    rate[np.logical_not(good_intervals)] = 0

    try:
        gtitable = lchdulist[gtistring].data
        gti_list = np.array([[a, b]
                             for a, b in zip(gtitable.field('START'),
                                             gtitable.field('STOP'))],
                            dtype=np.longdouble)
    except:
        gti_list = create_gti_from_condition(time, good_intervals)

    lchdulist.close()

    lc = Lightcurve(time=time, counts=rate, err=rate_e, gti=gti_list,
                    mjdref=mjdref.mjd)

    lc.instr = instr
    lc.header = lchdulist[ratehdu].header.tostring()

    logging.info('Saving light curve to %s' % outfile)
    save_lcurve(lc, outfile)
    return [outfile]


def lcurve_from_txt(txt_file, outfile=None,
                    noclobber=False, outdir=None,
                    mjdref=None, gti=None):
    """
    Load a lightcurve from a text file.

    Parameters
    ----------
    txt_file : str
        File name of the input light curve in text format. Assumes two columns:
        time, counts. Times are seconds from MJDREF 55197.00076601852 (NuSTAR)
        if not otherwise specified.

    Returns
    -------
    outfile : [str]
        Returned as a list with a single element for consistency with
        `lcurve_from_events`

    Other Parameters
    ----------------
    outfile : str
        Output file name
    noclobber : bool
        If True, do not overwrite existing files
    mjdref : float, default 55197.00076601852
        the MJD time reference
    gti : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good Time Intervals
    """
    import numpy as np
    if mjdref is None:
        mjdref = np.longdouble('55197.00076601852')

    outfile = assign_value_if_none(outfile, hen_root(txt_file) + '_lc')
    outfile = outfile.replace(HEN_FILE_EXTENSION, '') + HEN_FILE_EXTENSION

    outdir = assign_value_if_none(
        outdir, os.path.dirname(os.path.abspath(txt_file)))

    _, outfile = os.path.split(outfile)
    mkdir_p(outdir)
    outfile = os.path.join(outdir, outfile)

    if noclobber and os.path.exists(outfile):
        warnings.warn('File exists, and noclobber option used. Skipping')
        return [outfile]

    time, counts = np.genfromtxt(txt_file, delimiter=' ', unpack=True)
    time = np.array(time, dtype=np.longdouble)
    counts = np.array(counts, dtype=np.float)

    lc = Lightcurve(time=time, counts=counts, gti=gti,
                    mjdref=mjdref)

    lc.instr = 'EXTERN'

    logging.info('Saving light curve to %s' % outfile)
    save_lcurve(lc, outfile)
    return [outfile]


def _baseline_lightcurves(lcurves, outroot, p, lam):
    outroot_save = outroot
    for i, f in enumerate(lcurves):
        if outroot is None:
            outroot = hen_root(f) + '_lc_baseline'
        else:
            outroot = outroot_save + '_{}'.format(i)
        ftype, lc = get_file_type(f)
        baseline = lc.baseline(p, lam)
        lc.base = baseline
        save_lcurve(lc, outroot + HEN_FILE_EXTENSION)


def _wrap_lc(args):
    f, kwargs = args
    try:
        return lcurve_from_events(f, **kwargs)
    except Exception as e:
        warnings.warn("HENlcurve exception: {0}".format(str(e)))
        raise
        return []


def _wrap_txt(args):
    f, kwargs = args
    try:
        return lcurve_from_txt(f, **kwargs)
    except Exception as e:
        warnings.warn("HENlcurve exception: {0}".format(str(e)))
        return []


def _wrap_fits(args):
    f, kwargs = args
    try:
        return lcurve_from_fits(f, **kwargs)
    except Exception as e:
        warnings.warn("HENlcurve exception: {0}".format(str(e)))
        return []


def main(args=None):
    """Main function called by the `HENlcurve` command line script."""
    import argparse
    from multiprocessing import Pool

    description = ('Create lightcurves starting from event files. It is '
                   'possible to specify energy or channel filtering options')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of files", nargs='+')

    parser.add_argument("-b", "--bintime", type=float, default=1,
                        help="Bin time; if negative, negative power of 2")
    parser.add_argument("--safe-interval", nargs=2, type=float,
                        default=[0, 0],
                        help="Interval at start and stop of GTIs used" +
                        " for filtering")
    parser.add_argument("--pi-interval", type=int, default=[-1, -1],
                        nargs=2,
                        help="PI interval used for filtering")
    parser.add_argument('-e', "--e-interval", type=float, default=[-1, -1],
                        nargs=2,
                        help="Energy interval used for filtering")
    parser.add_argument("-s", "--scrunch",
                        help="Create scrunched light curve (single channel)",
                        default=False,
                        action="store_true")
    parser.add_argument("-j", "--join",
                        help="Create joint light curve (multiple channels)",
                        default=False,
                        action="store_true")
    parser.add_argument("-g", "--gti-split",
                        help="Split light curve by GTI",
                        default=False,
                        action="store_true")
    parser.add_argument("--minlen",
                        help="Minimum length of acceptable GTIs (default:4)",
                        default=4, type=float)
    parser.add_argument("--ignore-gtis",
                        help="Ignore GTIs",
                        default=False,
                        action="store_true")
    parser.add_argument("-d", "--outdir", type=str, default=None,
                        help='Output directory')
    parser.add_argument("-o", "--outfile", type=str, default=None,
                        help='Output file name')
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
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')
    parser.add_argument("--noclobber", help="Do not overwrite existing files",
                        default=False, action='store_true')
    parser.add_argument("--fits-input",
                        help="Input files are light curves in FITS format",
                        default=False, action='store_true')
    parser.add_argument("--txt-input",
                        help="Input files are light curves in txt format",
                        default=False, action='store_true')

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'
    bintime = args.bintime

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='HENlcurve.log', level=numeric_level,
                        filemode='w')

    safe_interval = args.safe_interval
    pi_interval = np.array(args.pi_interval)
    e_interval = np.array(args.e_interval)

    # ------ Use functools.partial to wrap lcurve* with relevant keywords---
    if args.fits_input:
        wrap_fun = _wrap_fits
        argdict = {"noclobber": args.noclobber}
    elif args.txt_input:
        wrap_fun = _wrap_txt
        argdict = {"noclobber": args.noclobber}
    else:
        wrap_fun = _wrap_lc
        argdict = {"noclobber": args.noclobber, "safe_interval": safe_interval,
                   "pi_interval": pi_interval,
                   "e_interval": e_interval,
                   "min_length": args.minlen,
                   "gti_split": args.gti_split,
                   "ignore_gtis": args.ignore_gtis,
                   "bintime": bintime, "outdir": args.outdir}

    arglist = [[f, argdict.copy()] for f in args.files]
    na = len(arglist)
    outfile = args.outfile
    if outfile is not None:
        outname = os.path.splitext(outfile)[0]
        for i in range(na):
            if na > 1:
                outname = outfile + "_{0}".format(i)
            arglist[i][1]['outfile'] = outname

    # -------------------------------------------------------------------------
    outfiles = []

    if os.name == 'nt' or args.nproc == 1:
        for a in arglist:
            outfiles.append(wrap_fun(a))
    else:
        pool = Pool(processes=args.nproc)
        for i in pool.imap_unordered(wrap_fun, arglist):
            outfiles.append(i)
        pool.close()

    logging.debug(outfiles)

    if args.scrunch:
        scrunch_lightcurves(outfiles)

    if args.join:
        join_lightcurves(outfiles)


def scrunch_main(args=None):
    """Main function called by the `HENscrunchlc` command line script."""
    import argparse
    description = \
        'Sum lightcurves from different instruments or energy ranges'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("-o", "--out", type=str,
                        default="out_scrlc" + HEN_FILE_EXTENSION,
                        help='Output file')
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
    logging.basicConfig(filename='HENscrunchlc.log', level=numeric_level,
                        filemode='w')

    scrunch_lightcurves(files, args.out)


def baseline_main(args=None):
    """Main function called by the `HENbaselinesub` command line script."""
    import argparse
    description = \
        ('Subtract a baseline from the lightcurve using the Asymmetric Least '
         'Squares algorithm. The two parameters p and lambda control the '
         'asymmetry and smoothness of the baseline. See below for details.')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("-o", "--out", type=str, default=None,
                        help='Output file')
    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')
    parser.add_argument("-p", '--asymmetry', type=float,
                        help='"asymmetry" parameter. Smaller values make the '
                             'baseline more "horizontal". Typically '
                             '0.001 < p < 0.1, but not necessarily.',
                        default=0.01)
    parser.add_argument("-l", "--lam", type=float,
                        help='lambda, or "smoothness", parameter. Larger'
                        ' values make the baseline stiffer. Typically '
                        '1e2 < lam < 1e9',
                        default=1e5)

    args = parser.parse_args(args)
    files = args.files

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='HENscrunchlc.log', level=numeric_level,
                        filemode='w')

    _baseline_lightcurves(files, args.out, args.asymmetry, args.lam)
