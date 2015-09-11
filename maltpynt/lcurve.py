# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Light curve-related functions."""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
from .base import mp_root, create_gti_mask, cross_gtis, mkdir_p
from .base import contiguous_regions, calc_countrate, gti_len
from .io import load_events, load_lcurve, save_lcurve
from .io import MP_FILE_EXTENSION, high_precision_keyword_read
import os
import logging


def lcurve(event_list,
           bin_time,
           start_time=None,
           stop_time=None,
           centertime=True):
    """From a list of event times, estract a lightcurve.

    Parameters
    ----------
    event_list : array-like
        Times of arrival of events
    bin_time : float
        Binning time of the light curve

    Returns
    -------
    time : array-like
        The time bins of the light curve
    lc : array-like
        The light curve

    Other Parameters
    ----------------
    start_time : float
        Initial time of the light curve
    stop_time : float
        Stop time of the light curve
    centertime: bool
        If False, time is the start of the bin. Otherwise, the center
    """
    if start_time is None:
        logging.warning("lcurve: Changing start time")
        start_time = np.floor(event_list[0])
    if stop_time is None:
        logging.warning("lcurve: Changing stop time")
        stop_time = np.ceil(event_list[-1])

    logging.debug("lcurve: Time limits: %g -- %g" %
                  (start_time, stop_time))

    new_event_list = event_list[event_list >= start_time]
    new_event_list = new_event_list[new_event_list <= stop_time]
    # To compute the histogram, the times array must specify the bin edges.
    # therefore, if nbin is the length of the lightcurve, times will have
    # nbin + 1 elements
    new_event_list = ((new_event_list - start_time) / bin_time).astype(int)
    times = np.arange(start_time, stop_time, bin_time)
    lc = np.bincount(new_event_list, minlength=len(times))
    logging.debug("lcurve: Length of the lightcurve: %g" % len(times))
    logging.debug("Times, kind: %s, %s" % (repr(times), type(times[0])))
    logging.debug("Lc, kind: %s, %s" % (repr(lc), type(lc[0])))
    logging.debug("bin_time, kind: %s, %s" % (repr(bin_time), type(bin_time)))
    if centertime:
        times = times + bin_time / 2.
    return times, lc.astype(np.float)


def join_lightcurves(lcfilelist, outfile='out_lc' + MP_FILE_EXTENSION):
    """Join light curves from different files.

    Light curves from different instruments are put in different channels.

    Parameters
    ----------
    lcfilelist :
    outfile :

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
    lcdts = [lcdata['dt'] for lcdata in lcdatas]
    # Find unique elements. If multiple bin times are used, throw an exception
    lcdts = list(set(lcdts))
    if len(lcdts) > 1:
        raise Exception('Light curves must have same dt for scrunching')

    instrs = [lcdata['Instr'] for lcdata in lcdatas]
    # Find unique elements. A lightcurve will be produced for each instrument
    instrs = list(set(instrs))
    outlcs = {}
    times = {}
    lcs = {}
    gtis = {}
    for instr in instrs:
        outlcs[instr] = {'dt': lcdts[0], 'Tstart': 1e32, 'Tstop': -11,
                         'MJDref': lcdatas[0]['MJDref'], 'source_ctrate': 0,
                         'total_ctrate': 0}
        times[instr] = []
        lcs[instr] = []
        gtis[instr] = []
    # -------------------------------------------------------

    for lcdata in lcdatas:
        tstart = lcdata['Tstart']
        tstop = lcdata['Tstop']
        if outlcs[instr]['Tstart'] > tstart:
            outlcs[instr]['Tstart'] = tstart
        if outlcs[instr]['Tstop'] < tstop:
            outlcs[instr]['Tstop'] = tstop

        time = lcdata['time']
        lc = lcdata['lc']
        gti = lcdata['GTI']
        instr = lcdata['Instr']
        times[instr].extend(time)
        lcs[instr].extend(lc)
        gtis[instr].extend(gti)

        goodlen = gti_len(gti)
        outlcs[instr]['source_ctrate'] += lcdata['source_ctrate'] * goodlen
        outlcs[instr]['total_ctrate'] += lcdata['total_ctrate'] * goodlen

    for instr in instrs:
        gti = np.array(gtis[instr])
        outlcs[instr]['time'] = np.array(times[instr])
        outlcs[instr]['lc'] = np.array(lcs[instr])
        outlcs[instr]['GTI'] = gti
        outlcs[instr]['Instr'] = instr

        goodlen = gti_len(gti)

        outlcs[instr]['source_ctrate'] /= goodlen
        outlcs[instr]['total_ctrate'] /= goodlen

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


def scrunch_lightcurves(lcfilelist, outfile='out_scrlc'+MP_FILE_EXTENSION,
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
    gti_lists = [lcdata[inst]['GTI'] for inst in instrs]
    gti = cross_gtis(gti_lists)
    # Determine limits
    time0 = lcdata[instrs[0]]['time']
    mask = create_gti_mask(time0, gti)

    time0 = time0[mask]
    lc0 = lcdata[instrs[0]]['lc']
    lc0 = lc0[mask]

    for inst in instrs[1:]:
        time1 = lcdata[inst]['time']
        mask = create_gti_mask(time1, gti)
        time1 = time1[mask]
        assert np.all(time0 == time1), \
            'Something is not right with gti filtering'
        lc = lcdata[inst]['lc']
        lc0 += lc[mask]

    out = lcdata[instrs[0]].copy()
    out['lc'] = lc0
    out['time'] = time0
    out['dt'] = lcdata[instrs[0]]['dt']
    out['GTI'] = gti

    out['Instr'] = ",".join(instrs)

    out['source_ctrate'] = np.sum([lcdata[i]['source_ctrate'] for i in instrs])
    out['total_ctrate'] = np.sum([lcdata[i]['total_ctrate'] for i in instrs])

    logging.info('Saving scrunched light curve to %s' % outfile)
    save_lcurve(out, outfile)

    return time0, lc0, gti


def filter_lc_gtis(time, lc, gti, safe_interval=None, delete=False,
                   min_length=0, return_borders=False):
    """Filter a light curve for GTIs.

    Parameters
    ----------
    time : array-like
        The time bins of the light curve
    lc : array-like
        The light curve
    gti : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        Good Time Intervals

    Returns
    -------
    time : array-like
        The time bins of the light curve
    lc : array-like
        The output light curve
    newgtis : [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
        The output Good Time Intervals
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
    mask, newgtis = create_gti_mask(time, gti, return_new_gtis=True,
                                    safe_interval=safe_interval,
                                    min_length=min_length)

    nomask = np.logical_not(mask)

    if delete:
        time = time[mask]
        lc = lc[mask]
    else:
        lc[nomask] = 0

    if return_borders:
        mask = create_gti_mask(time, newgtis)
        borders = contiguous_regions(mask)
        return time, lc, newgtis, borders
    else:
        return time, lc, newgtis


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
    if (outfile is not None) and (outdir is not None):
        raise Exception('Please specify only one between outdir and outfile')

    logging.info("Loading file %s..." % f)
    evdata = load_events(f)
    logging.info("Done.")

    if bintime < 0:
        bintime = 2 ** (bintime)
    bintime = np.longdouble(bintime)

    tag = ''
    out = {}
    tstart = evdata['Tstart']
    tstop = evdata['Tstop']
    events = evdata['time']
    instr = evdata['Instr']
    mjdref = evdata['MJDref']

    if instr == 'PCA':
        pcus = evdata['PCU']
    gtis = evdata['GTI']
    if ignore_gtis:
        gtis = np.array([[tstart, tstop]])

    out['MJDref'] = mjdref
    # make tstart and tstop multiples of bin times since MJDref
    tstart = np.ceil(tstart / bintime, dtype=np.longdouble) * bintime
    tstop = np.floor(tstop / bintime, dtype=np.longdouble) * bintime

    # First of all, calculate total count rate (no filtering applied)
    tot_time, tot_lc = lcurve(events, bintime, start_time=tstart,
                              stop_time=tstop)

    tot_time, tot_lc, newgtis, tot_borders = \
        filter_lc_gtis(tot_time, tot_lc, gtis,
                       safe_interval=safe_interval,
                       delete=False,
                       min_length=min_length,
                       return_borders=True)

    out['total_ctrate'] = calc_countrate(tot_time, tot_lc, bintime=bintime)

    # Then, apply filters
    if pi_interval is not None and np.all(np.array(pi_interval) > 0):
        pis = evdata['PI']
        good = np.logical_and(pis > pi_interval[0],
                              pis <= pi_interval[1])
        events = events[good]
        tag = '_PI%g-%g' % (pi_interval[0], pi_interval[1])
        out['PImin'] = e_interval[0]
        out['PImax'] = e_interval[0]
    elif e_interval is not None and np.all(np.array(e_interval) > 0):
        try:
            es = evdata['E']
        except:
            raise \
                ValueError("No energy information is present in the file." +
                           " Did you run MPcalibrate?")

        good = np.logical_and(es > e_interval[0],
                              es <= e_interval[1])
        events = events[good]
        tag = '_E%g-%g' % (e_interval[0], e_interval[1])
        out['Emin'] = e_interval[0]
        out['Emax'] = e_interval[1]

    if outfile is None:
        outfile = mp_root(f) + tag + '_lc' + MP_FILE_EXTENSION
    else:
        outfile = \
            outfile.replace(MP_FILE_EXTENSION, '') + MP_FILE_EXTENSION
    if noclobber and os.path.exists(outfile):
        print('File exists, and noclobber option used. Skipping')
        return [outfile]

    time, lc = lcurve(events, bintime, start_time=tstart,
                      stop_time=tstop)

    time, lc, newgtis, borders = \
        filter_lc_gtis(time, lc, gtis,
                       safe_interval=safe_interval,
                       delete=False,
                       min_length=min_length,
                       return_borders=True)

    if len(newgtis) == 0:
        print("No GTIs above min_length ({0}s) found.".format(min_length))
        return

    assert np.all(borders == tot_borders), \
        'Borders do not coincide: {0} {1}'.format(borders, tot_borders)

    out['source_ctrate'] = calc_countrate(time, lc, gtis=newgtis,
                                          bintime=bintime)

    if outdir is not None:
        _, f = os.path.split(f)
        mkdir_p(outdir)
        f = os.path.join(outdir, f)

    if gti_split:
        outfiles = []
        logging.debug(borders)
        for ib, b in enumerate(borders):
            local_tag = tag + '_gti%d' % ib
            outfile = mp_root(f) + local_tag + '_lc' + MP_FILE_EXTENSION
            if noclobber and os.path.exists(outfile):
                print('File exists, and noclobber option used. Skipping')
                outfiles.append(outfile)

            logging.debug(b)
            local_out = out.copy()
            local_out['lc'] = lc[b[0]:b[1]]
            local_out['time'] = time[b[0]:b[1]]
            local_out['dt'] = bintime
            local_gti = np.array([[time[b[0]], time[b[1]-1]]])
            local_out['GTI'] = local_gti
            local_out['Tstart'] = time[b[0]]
            local_out['Tstop'] = time[b[1]-1]
            local_out['Instr'] = instr
            local_out['source_ctrate'] = calc_countrate(time[b[0]:b[1]],
                                                        lc[b[0]:b[1]],
                                                        bintime=bintime)
            local_out['total_ctrate'] = calc_countrate(tot_time[b[0]:b[1]],
                                                       tot_lc[b[0]:b[1]],
                                                       bintime=bintime)
            if instr == 'PCA':
                local_out['nPCUs'] = len(set(pcus))

            logging.info('Saving light curve to %s' % outfile)
            save_lcurve(local_out, outfile)
            outfiles.append(outfile)
    else:
        out['lc'] = lc
        out['time'] = time
        out['dt'] = bintime
        out['GTI'] = newgtis
        out['Tstart'] = tstart
        out['Tstop'] = tstop
        out['Instr'] = instr
        if instr == 'PCA':
            out['nPCUs'] = len(set(pcus))

        logging.info('Saving light curve to %s' % outfile)
        save_lcurve(out, outfile)
        outfiles = [outfile]

    # For consistency in return value
    return outfiles


def lcurve_from_fits(fits_file, gtistring='GTI',
                     timecolumn='TIME', ratecolumn=None, ratehdu=1,
                     fracexp_limit=0.9, outfile=None,
                     noclobber=False):
    """
    Load a lightcurve from a fits file and save it in MaLTPyNT format.

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
    from .base import create_gti_from_condition

    if outfile is None:
        outfile = mp_root(fits_file) + '_lc'

    outfile = outfile.replace(MP_FILE_EXTENSION, '') + MP_FILE_EXTENSION

    if noclobber and os.path.exists(outfile):
        print('File exists, and noclobber option used. Skipping')
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
    try:
        timezero = high_precision_keyword_read(lchdulist[ratehdu].header,
                                                'TIMEZERO')
        # Sometimes timezero is "from tstart", sometimes it's an absolute time.
        # This tries to detect which case is this, and always consider it
        # referred to tstart
    except:
        timezero = 0

    # for lcurve light curves this should instead work
    if tunit == 'd':
        # TODO:
        # Check this. For now, I assume TD (JD - 2440000.5).
        # This is likely wrong
        timezero = Time(2440000.5 + timezero, scale='tdb', format='jd')
        tstart = Time(2440000.5 + tstart, scale='tdb', format='jd')
        tstop = Time(2440000.5 + tstop, scale='tdb', format='jd')
        if mjdref is None:
            # use NuSTAR defaulf MJDREF
            mjdref = Time(np.longdouble('55197.00076601852'), scale='tdb',
                          format='mjd')
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
        print('Assuming that TIMEDEL is the difference between the first two'
              'times of the light curve')
        dt = time[1] - time[0]

    # ----------------------------------------------------------------
    if ratecolumn is None:
        for i in ['RATE', 'RATE1', 'COUNTS']:
            if i in lctable.names:
                ratecolumn = i
                break
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

    good_intervals = np.logical_and(fracexp >= fracexp_limit, fracexp <= 1)
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

    out = {}
    out['lc'] = rate
    out['elc'] = rate_e
    out['time'] = time
    out['dt'] = dt
    out['GTI'] = gti_list
    out['Tstart'] = tstart
    out['Tstop'] = tstop
    out['Instr'] = instr

    out['MJDref'] = mjdref.value

    out['total_ctrate'] = calc_countrate(time, rate, gtis=gti_list,
                                         bintime=dt)
    out['source_ctrate'] = calc_countrate(time, rate, gtis=gti_list,
                                          bintime=dt)

    logging.info('Saving light curve to %s' % outfile)
    save_lcurve(out, outfile)
    return [outfile]


def lcurve_from_txt(txt_file, outfile=None,
                    noclobber=False):
    """
    Load a lightcurve from a text file.


    Parameters
    ----------
    txt_file : str
        File name of the input light curve in text format. Assumes two columns:
        time, counts. Times are seconds from MJDREF 55197.00076601852 (NuSTAR).

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
    """
    import numpy as np

    if outfile is None:
        outfile = mp_root(txt_file) + '_lc'
    outfile = outfile.replace(MP_FILE_EXTENSION, '') + MP_FILE_EXTENSION

    if noclobber and os.path.exists(outfile):
        print('File exists, and noclobber option used. Skipping')
        return [outfile]

    time, lc = np.genfromtxt(txt_file, delimiter=' ', unpack=True)
    time = np.array(time, dtype=np.longdouble)
    lc = np.array(lc, dtype=np.float)
    dt = time[1] - time[0]
    out = {}

    out['lc'] = lc
    out['time'] = time
    out['dt'] = dt
    gtis = np.array([[time[0] - dt / 2, time[-1] + dt / 2]])
    out['GTI'] = gtis
    out['Tstart'] = time[0] - dt / 2
    out['Tstop'] = time[-1] + dt / 2
    out['Instr'] = 'EXTERN'
    out['MJDref'] = np.longdouble('55197.00076601852')
    out['total_ctrate'] = calc_countrate(time, lc, gtis=gtis,
                                         bintime=dt)
    out['source_ctrate'] = calc_countrate(time, lc, gtis=gtis,
                                          bintime=dt)

    logging.info('Saving light curve to %s' % outfile)
    save_lcurve(out, outfile)
    return [outfile]


def _wrap_lc(args):
    f, kwargs = args
    return lcurve_from_events(f, **kwargs)


def _wrap_txt(args):
    f, kwargs = args
    return lcurve_from_txt(f, **kwargs)


def _wrap_fits(args):
    f, kwargs = args
    return lcurve_from_fits(f, **kwargs)


def main(args=None):
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
    logging.basicConfig(filename='MPlcurve.log', level=numeric_level,
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

    arglist = [[f, argdict] for f in args.files]
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
    import argparse
    description = \
        'Sum lightcurves from different instruments or energy ranges'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("files", help="List of files", nargs='+')
    parser.add_argument("-o", "--out", type=str,
                        default="out_scrlc" + MP_FILE_EXTENSION,
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
    logging.basicConfig(filename='MPscrunchlc.log', level=numeric_level,
                        filemode='w')

    scrunch_lightcurves(files, args.out)
