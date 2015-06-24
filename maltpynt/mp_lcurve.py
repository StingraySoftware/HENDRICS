# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
from .mp_base import mp_root, mp_create_gti_mask, mp_cross_gtis, mp_mkdir_p
from .mp_base import mp_contiguous_regions, mp_calc_countrate, mp_gti_len
from .mp_io import mp_load_events, mp_load_lcurve, mp_save_lcurve
from .mp_io import MP_FILE_EXTENSION
import os
import logging


def mp_lcurve(event_list,
              bin_time,
              start_time=None,
              stop_time=None,
              centertime=True):
    '''
    From a list of event times, estract a lightcurve

    Usage:
    times, lc = bin_events(event_list, bin_time)

    Optional keywords:
    start_time
    stop_time
    centertime: if False, time is teh start of the bin
    '''
    if start_time is None:
        logging.warning("mp_lcurve: Changing start time")
        start_time = np.floor(event_list[0])
    if stop_time is None:
        logging.warning("mp_lcurve: Changing stop time")
        stop_time = np.ceil(event_list[-1])
    logging.debug("mp_lcurve: Time limits: %g -- %g" %
                  (start_time, stop_time))

    new_event_list = event_list[event_list >= start_time]
    new_event_list = new_event_list[new_event_list <= stop_time]
    # To compute the histogram, the times array must specify the bin edges.
    # therefore, if nbin is the length of the lightcurve, times will have
    # nbin + 1 elements
    new_event_list = ((new_event_list - start_time) / bin_time).astype(int)
    times = np.arange(start_time, stop_time, bin_time)
    lc = np.bincount(new_event_list, minlength=len(times))
    logging.debug("mp_lcurve: Length of the lightcurve: %g" % len(times))
    logging.debug("Times, kind: %s, %s" % (repr(times), type(times[0])))
    logging.debug("Lc, kind: %s, %s" % (repr(lc), type(lc[0])))
    logging.debug("bin_time, kind: %s, %s" % (repr(bin_time), type(bin_time)))
    if centertime:
        times = times + bin_time / 2.
    return times, lc.astype(np.float)


def mp_join_lightcurves(lcfilelist, outfile='out_lc' + MP_FILE_EXTENSION):
    lcdatas = []
    for lfc in lcfilelist:
        logging.info("Loading file %s..." % lfc)
        lcdata = mp_load_lcurve(lfc)
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

        goodlen = mp_gti_len(gti)
        outlcs[instr]['source_ctrate'] += lcdata['source_ctrate'] * goodlen
        outlcs[instr]['total_ctrate'] += lcdata['total_ctrate'] * goodlen

    for instr in instrs:
        gti = np.array(gtis[instr])
        outlcs[instr]['time'] = np.array(times[instr])
        outlcs[instr]['lc'] = np.array(lcs[instr])
        outlcs[instr]['GTI'] = gti
        outlcs[instr]['Instr'] = instr

        goodlen = mp_gti_len(gti)

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
            mp_save_lcurve(outlcs[instr], os.path.join(dname, tag + fname))

    return outlcs


def mp_scrunch_lightcurves(lcfilelist, outfile='out_scrlc'+MP_FILE_EXTENSION,
                           save_joint=False):
    '''Create a single light curve from input light curves,
    regardless of the instrument'''
    if save_joint:
        lcdata = mp_join_lightcurves(lcfilelist)
    else:
        lcdata = mp_join_lightcurves(lcfilelist, outfile=None)

    instrs = list(lcdata.keys())
    gti_lists = [lcdata[inst]['GTI'] for inst in instrs]
    gti = mp_cross_gtis(gti_lists)
    # Determine limits
    time0 = lcdata[instrs[0]]['time']
    mask = mp_create_gti_mask(time0, gti)

    time0 = time0[mask]
    lc0 = lcdata[instrs[0]]['lc']
    lc0 = lc0[mask]

    for inst in instrs[1:]:
        time1 = lcdata[inst]['time']
        mask = mp_create_gti_mask(time1, gti)
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
    mp_save_lcurve(out, outfile)

    return time0, lc0, gti


def mp_filter_lc_gtis(time, lc, gti, safe_interval=None, delete=False,
                      min_length=0, return_borders=False):

    mask, newgtis = mp_create_gti_mask(time, gti, return_new_gtis=True,
                                       safe_interval=safe_interval,
                                       min_length=min_length)

#    # test if newgti-created mask coincides with mask
#    newmask = mp_create_gti_mask(time, newgtis, safe_interval=0)
#    print("Test: newly created gti is equivalent?", np.all(newmask == mask))

    nomask = np.logical_not(mask)

    if delete:
        time = time[mask]
        lc = lc[mask]
    else:
        lc[nomask] = 0

    if return_borders:
        mask = mp_create_gti_mask(time, newgtis)
        borders = mp_contiguous_regions(mask)
        return time, lc, newgtis, borders
    else:
        return time, lc, newgtis


def mp_lcurve_from_events(f, safe_interval=0,
                          pi_interval=None,
                          e_interval=None,
                          min_length=0,
                          gti_split=False,
                          ignore_gtis=False,
                          bintime=1.,
                          outdir=None,
                          outfile=None):
    if (outfile is not None) and (outdir is not None):
        raise Exception('Please specify only one between outdir and outfile')
    logging.info("Loading file %s..." % f)
    evdata = mp_load_events(f)
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
    time, lc = mp_lcurve(events, bintime, start_time=tstart,
                         stop_time=tstop)

    time, lc, newgtis = \
        mp_filter_lc_gtis(time, lc, gtis,
                          safe_interval=safe_interval,
                          delete=True,
                          min_length=min_length)

    out['total_ctrate'] = mp_calc_countrate(time, lc, bintime=bintime)

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
                ValueError("No energy information is present in the file."
                           + " Did you run MPcalibrate?")

        good = np.logical_and(es > e_interval[0],
                              es <= e_interval[1])
        events = events[good]
        tag = '_E%g-%g' % (e_interval[0], e_interval[1])
        out['Emin'] = e_interval[0]
        out['Emax'] = e_interval[0]

    time, lc = mp_lcurve(events, bintime, start_time=tstart,
                         stop_time=tstop)

    time, lc, newgtis, borders = \
        mp_filter_lc_gtis(time, lc, gtis,
                          safe_interval=safe_interval,
                          delete=False,
                          min_length=min_length,
                          return_borders=True)

    out['source_ctrate'] = mp_calc_countrate(time, lc, gtis=newgtis,
                                             bintime=bintime)

    if outdir is not None:
        _, f = os.path.split(f)
        mp_mkdir_p(outdir)
        f = os.path.join(outdir, f)

    # TODO: implement per-interval count rates
    if gti_split:
        outfiles = []
        logging.debug(borders)
        for ib, b in enumerate(borders):
            logging.debug(b)
            local_tag = tag + '_gti%d' % ib
            local_out = out.copy()
            local_out['lc'] = lc[b[0]:b[1]]
            local_out['time'] = time[b[0]:b[1]]
            local_out['dt'] = bintime
            local_out['GTI'] = [[time[b[0]], time[b[1]]]]
            local_out['Tstart'] = time[b[0]]
            local_out['Tstop'] = time[b[1]]
            local_out['Instr'] = instr
            if instr == 'PCA':
                local_out['nPCUs'] = len(set(pcus))

            outfile = mp_root(f) + local_tag + '_lc' + MP_FILE_EXTENSION
            logging.info('Saving light curve to %s' % outfile)
            mp_save_lcurve(local_out, outfile)
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

        if outfile is None:
            outfile = mp_root(f) + tag + '_lc' + MP_FILE_EXTENSION
        else:
            outfile = \
                outfile.replace(MP_FILE_EXTENSION, '') + MP_FILE_EXTENSION
        logging.info('Saving light curve to %s' % outfile)
        mp_save_lcurve(out, outfile)
        outfiles = [outfile]

    # For consistency in return value
    return outfiles


def _high_precision_keyword_read(hdr, keyword):
    try:
        value = np.longdouble(hdr[keyword])
    except:
        if len(keyword) == 8:
            keyword = keyword[:7]
        value = np.longdouble(hdr[keyword + 'I'])
        value += np.longdouble(hdr[keyword + 'F'])

    return value


def mp_lcurve_from_fits(fits_file, gtistring='GTI',
                        timecolumn='TIME', ratecolumn=None, ratehdu=1,
                        fracexp_limit=0.9, outfile=None):
    '''
    Load a lightcurve from a fits file.

    Outputs a light curve file in MaLTPyNT format
    '''
    logging.warning(
        '''WARNING! FITS light curve handling is still under testing.
        Absolute times might be incorrect''')
    # TODO:
    # treat consistently TDB, UTC, TAI, etc. This requires some documentation
    # reading. For now, we assume TDB
    from astropy.io import fits as pf
    from astropy.time import Time
    import numpy as np
    from .mp_base import mp_create_gti_from_condition

    lchdulist = pf.open(fits_file)
    lctable = lchdulist[ratehdu].data

    # Units of header keywords
    tunit = lchdulist[ratehdu].header['TIMEUNIT']

    try:
        mjdref = _high_precision_keyword_read(lchdulist[ratehdu].header,
                                              'MJDREF')
        mjdref = Time(mjdref, scale='tdb', format='mjd')
    except:
        mjdref = None

    # ----------------------------------------------------------------
    # Trying to comply with all different formats of fits light curves.
    # It's a madness...
    try:
        tstart = _high_precision_keyword_read(lchdulist[ratehdu].header,
                                              'TSTART')
        tstop = _high_precision_keyword_read(lchdulist[ratehdu].header,
                                             'TSTOP')
    except:
        raise(Exception('TSTART and TSTOP need to be specified'))

    # For nulccorr lcs this whould work
    try:
        timezero = _high_precision_keyword_read(lchdulist[ratehdu].header,
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
        dt = _high_precision_keyword_read(lchdulist[ratehdu].header,
                                          'TIMEDEL')
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
        gti_list = mp_create_gti_from_condition(time, good_intervals)

    lchdulist.close()

    out = {}
    out['lc'] = rate
    out['time'] = time
    out['dt'] = dt
    out['GTI'] = gti_list
    out['Tstart'] = tstart
    out['Tstop'] = tstop
    out['Instr'] = 'EXTERN'

    out['MJDref'] = mjdref.value

    out['total_ctrate'] = mp_calc_countrate(time, rate, gtis=gti_list,
                                            bintime=dt)
    out['source_ctrate'] = mp_calc_countrate(time, rate, gtis=gti_list,
                                             bintime=dt)

    if outfile is None:
        outfile = mp_root(fits_file) + '_lc'

    outfile = outfile.replace(MP_FILE_EXTENSION, '') + MP_FILE_EXTENSION

    logging.info('Saving light curve to %s' % outfile)
    mp_save_lcurve(out, outfile)
    return [outfile]


def mp_lcurve_from_txt(txt_file, outfile=None):
    '''
    Load a lightcurve from a text file.

    Assumes two columns: time, counts. Times are by default seconds from
    MJDREF 55197.00076601852 (NuSTAR).
    '''
    import numpy as np

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
    out['total_ctrate'] = mp_calc_countrate(time, lc, gtis=gtis,
                                            bintime=dt)
    out['source_ctrate'] = mp_calc_countrate(time, lc, gtis=gtis,
                                             bintime=dt)

    if outfile is None:
        outfile = mp_root(txt_file) + '_lc'
    outfile = outfile.replace(MP_FILE_EXTENSION, '') + MP_FILE_EXTENSION
    logging.info('Saving light curve to %s' % outfile)
    mp_save_lcurve(out, outfile)
    return [outfile]


if __name__ == "__main__":
    import sys
    import subprocess as sp

    print('Calling script...')

    args = sys.argv[1:]

    sp.check_call(['MPlcurve'] + args)
