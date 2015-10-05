# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to simulate data and produce a fake event file."""

from __future__ import (absolute_import, division,
                        print_function)

import numpy as np
import numpy.random as ra
import os
import logging
import warnings
from .io import get_file_format, load_lcurve
from .base import _empty, _assign_value_if_none
from .lcurve import lcurve_from_fits


def fake_events_from_lc(
        times, lc, use_spline=False, bin_time=None):
    """Create events from a light curve.

    Parameters
    ----------
    times : array-like
        the center time of each light curve bin
    lc : array-like
        light curve, in units of counts/bin

    Returns
    -------
    event_list : array-like
        Simulated arrival times
    """
    try:
        import scipy.interpolate as sci
    except:
        if use_spline:
            warnings.warn("Scipy not available. ",
                          "use_spline option cannot be used.")
            use_spline = False

    bin_time = _assign_value_if_none(bin_time, times[1] - times[0])
    n_bin = len(lc)

    bin_start = 0

    maxlc = np.max(lc)
    intlc = maxlc * n_bin

    n_events_predict = int(intlc + 10 * np.sqrt(intlc))

    # Max number of events per chunk must be < 100000
    events_per_bin_predict = n_events_predict / n_bin
    if use_spline:
        max_bin = np.max([4, 1000000 / events_per_bin_predict])
        logging.debug("Using splines")
    else:
        max_bin = np.max([4, 5000000 / events_per_bin_predict])

    ev_list = np.zeros(n_events_predict)

    nev = 0

    while bin_start < n_bin:
        t0 = times[bin_start]
        bin_stop = min([bin_start + max_bin, n_bin + 1])
        lc_filt = lc[bin_start:bin_stop]
        t_filt = times[bin_start:bin_stop]
        logging.debug("{} {}".format(t_filt[0] - bin_time / 2,
                                     t_filt[-1] + bin_time / 2))

        length = t_filt[-1] - t_filt[0]
        n_bin_filt = len(lc_filt)
        n_to_simulate = n_bin_filt * max(lc_filt)
        safety_factor = 10
        if n_to_simulate > 1000:
            safety_factor = 3.

        n_to_simulate += safety_factor * np.sqrt(n_to_simulate)
        n_to_simulate = int(np.ceil(n_to_simulate))

        n_predict = ra.poisson(np.sum(lc_filt))

        random_ts = ra.uniform(t_filt[0] - bin_time / 2,
                               t_filt[-1] + bin_time / 2, n_to_simulate)

        logging.debug(random_ts[random_ts < 0])

        random_amps = ra.uniform(0, max(lc_filt), n_to_simulate)
        if use_spline:
            # print("Creating spline representation")
            lc_spl = sci.splrep(t_filt, lc_filt, s=np.longdouble(0), k=1)

            pts = sci.splev(random_ts, lc_spl)
        else:
            rough_bins = np.rint((random_ts - t0) / bin_time)
            rough_bins = rough_bins.astype(int)

            pts = lc_filt[rough_bins]

        good = random_amps < pts
        len1 = len(random_ts)
        random_ts = random_ts[good]
        len2 = len(random_ts)
        logging.debug("{0} Events generated".format(len1))
        logging.debug("{0} Events rejected".format(len1 - len2))
        random_ts = random_ts[:n_predict]
        random_ts.sort()
        new_nev = len(random_ts)
        ev_list[nev:nev + new_nev] = random_ts[:]
        nev += new_nev
        logging.debug(
            "{0} good events created ({1} ev/s)".format(new_nev,
                                                        new_nev / length))
        bin_start += max_bin

    # Discard all zero entries at the end!
    ev_list = ev_list[:nev]
    ev_list.sort()
    return ev_list


def _max_dead_timed_events(ev_list, deadtime):
    from .lcurve import lcurve
    try:
        time, lc = lcurve(ev_list, 1)
        return max(lc) * deadtime
    except:
        return 1


def filter_for_deadtime(ev_list, deadtime, bkg_ev_list=None,
                        dt_sigma=None, paralyzable=False,
                        additional_data=None, return_all=False):
    """Filter an event list for a given dead time.

    Parameters
    ----------
    ev_list : array-like
        The event list
    deadtime: float
        The (mean, if not constant) value of deadtime

    Returns
    -------
    new_ev_list : array-like
        The filtered event list
    additional_output : dict
        Object with all the following attributes. Only returned if
        `return_all` is True
    uf_events : array-like
        Unfiltered event list (events + background)
    is_event : array-like
        Boolean values; True if event, False if background
    deadtime : array-like
        Dead time values
    bkg : array-like
        The filtered background event list
    mask : array-like, optional
        The mask that filters the input event list and produces the output
        event list.

    Other Parameters
    ----------------
    bkg_ev_list : array-like
        A background event list that affects dead time
    dt_sigma : float
        The standard deviation of a non-constant dead time around deadtime.
    return_all : bool
        If True, return the mask that filters the input event list to obtain
        the output event list.

    """
    additional_output = _empty()

    if deadtime <= 0.:
        return np.copy(ev_list)

    # Create the total lightcurve, and a "kind" array that keeps track
    # of the events classified as "signal" (True) and "background" (False)
    if bkg_ev_list is not None:
        tot_ev_list = np.append(ev_list, bkg_ev_list)
        ev_kind = np.append(np.ones(len(ev_list), dtype=bool),
                            np.zeros(len(bkg_ev_list), dtype=bool))
        order = np.argsort(tot_ev_list)
        tot_ev_list = tot_ev_list[order]
        ev_kind = ev_kind[order]
        del order
    else:
        tot_ev_list = ev_list
        ev_kind = np.ones(len(ev_list), dtype=bool)

    mask = np.ones(len(tot_ev_list), dtype=bool)
    nevents = len(tot_ev_list)
    all_ev_kind = ev_kind.copy()

    if dt_sigma is not None:
        deadtime_values = ra.normal(deadtime, dt_sigma, nevents)
    else:
        deadtime_values = np.zeros(nevents) + deadtime

    dead_time_end = tot_ev_list + deadtime_values

    initial_len = len(tot_ev_list)
    saved_mask = mask.copy()
    if paralyzable:
        bad = dead_time_end[:-1] > tot_ev_list[1:]
        # Easy: paralyzable case. Here, events coming during dead time produce
        # more dead time. So...
        mask[1:][bad] = False
        tot_ev_list = tot_ev_list[mask]
        ev_kind = ev_kind[mask]
        saved_mask[np.logical_not(mask)] = False
    else:
        # Otherwise, it is a little trickier. An event is filtered if it comes
        # during dead time AND the previous event was valid. We need to iterate
        count = 1
        max_lookback = np.ceil(
            _max_dead_timed_events(tot_ev_list, np.max(deadtime_values)))
        max_lookback = int(np.min([max_lookback, len(tot_ev_list) - 1]))

        while True:
            mask_2 = np.zeros_like(mask)

            before_deadtime = \
                dead_time_end[:-max_lookback] > tot_ev_list[max_lookback:]
            mask_2[max_lookback:] = before_deadtime
            # flakes complains, but using "is True" would compare the entire
            # array, not the single elements, giving always false
            bad = np.logical_and(mask_2[max_lookback:] == True,
                                 mask_2[:-max_lookback] == 0)

            mask[max_lookback:] = np.logical_not(bad)

            if np.all(mask):
                if max_lookback == 1:
                    break
                max_lookback -= 1
                logging.debug(
                    'filter_for_deadtime, lookback {}'.format(max_lookback))
                continue

            tot_ev_list = tot_ev_list[mask]
            ev_kind = ev_kind[mask]
            deadtime_values = deadtime_values[mask]
            sm = saved_mask[saved_mask]
            sm[np.logical_not(mask)] = False
            saved_mask[saved_mask] = sm
            dead_time_end = tot_ev_list + deadtime_values
            len1 = len(mask)
            mask = mask[mask]
            len2 = len(mask)
            logging.debug(
                'filter_for_deadtime, pass '
                '{0}: {1}/{2} new events rejected'.format(count, len1 - len2,
                                                          len1))
            logging.debug(
                'filter_for_deadtime, pass '
                '{0}: {1} new events rejected'.format(count, tot_ev_list))
            count += 1

    final_len = len(tot_ev_list)
    logging.info(
        'filter_for_deadtime: '
        '{0}/{1} events rejected'.format(initial_len - final_len,
                                         initial_len))
    retval = tot_ev_list[ev_kind]
    if return_all:
        additional_output.uf_events = tot_ev_list
        additional_output.is_event = ev_kind
        additional_output.deadtime = deadtime_values
        additional_output.mask = saved_mask[all_ev_kind]
        additional_output.bkg = tot_ev_list[np.logical_not(ev_kind)]
        retval = [retval, additional_output]

    return retval


def generate_fake_fits_observation(event_list=None, filename=None, pi=None,
                                   instr='FPMA', gti=None, tstart=None,
                                   tstop=None,
                                   mjdref=55197.00076601852,
                                   livetime=None, additional_columns={}):
    """Generate fake NuSTAR data.

    Takes an event list (as a list of floats)
    All inputs are None by default, and can be set during the call.

    Parameters
    ----------
    event_list : list-like
        List of event arrival times, in seconds from mjdref. If left None, 1000
        random events will be generated, for a total length of 1025 s or the
        difference between tstop and tstart.
    filename : str
        Output file name

    Returns
    -------
    hdulist : FITS hdu list
        FITS hdu list of the output file

    Other Parameters
    ----------------
    mjdref : float
        Reference MJD. Default is 55197.00076601852 (NuSTAR)
    pi : list-like
        The PI channel of each event
    tstart : float
        Start of the observation (s from mjdref)
    tstop : float
        End of the observation (s from mjdref)
    instr : str
        Name of the instrument. Default is 'FPMA'
    livetime : float
        Total livetime. Default is tstop - tstart
    """
    from astropy.io import fits
    import numpy.random as ra

    if event_list is None:
        tstart = _assign_value_if_none(tstart, 8e+7)
        tstop = _assign_value_if_none(tstop, tstart + 1025)
        event_list = sorted(ra.uniform(tstart, tstop, 1000))

    pi = _assign_value_if_none(pi, ra.randint(0, 1024, len(event_list)))

    assert len(event_list) == len(pi), \
        "Event list and pi must be of the same length"

    tstart = _assign_value_if_none(tstart, np.floor(event_list[0]))
    tstop = _assign_value_if_none(tstop, np.ceil(event_list[-1]))
    gti = _assign_value_if_none(gti, np.array([[tstart, tstop]]))
    filename = _assign_value_if_none(filename, 'events.evt')
    livetime = _assign_value_if_none(livetime, tstop - tstart)

    assert livetime <= tstop - tstart, \
        'Livetime must be equal or smaller than tstop - tstart'

    # Create primary header
    prihdr = fits.Header()
    prihdr['OBSERVER'] = 'Edwige Bubble'
    prihdr['TELESCOP'] = ('NuSTAR  ', 'Telescope (mission) name')
    prihdr['INSTRUME'] = ('FPMA    ', 'Instrument name')
    prihdu = fits.PrimaryHDU(header=prihdr)

    # Write events to table
    col1 = fits.Column(name='TIME', format='1D', array=event_list)
    col2 = fits.Column(name='PI', format='1J', array=pi)

    allcols = [col1, col2]

    for c in additional_columns.keys():
        col = fits.Column(name=c, array=additional_columns[c]["data"],
                          format=additional_columns[c]["format"])
        allcols.append(col)

    cols = fits.ColDefs(allcols)
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name = 'EVENTS'

    # ---- Fake lots of information ----
    tbheader = tbhdu.header
    tbheader['OBSERVER'] = 'Edwige Bubble'
    tbheader['COMMENT'] = ("FITS (Flexible Image Transport System) format is"
                           " defined in 'Astronomy and Astrophysics', volume"
                           " 376, page 359; bibcode: 2001A&A...376..359H")
    tbheader['TELESCOP'] = ('NuSTAR  ', 'Telescope (mission) name')
    tbheader['INSTRUME'] = (instr, 'Instrument name')
    tbheader['OBS_ID'] = ('00000000001', 'Observation ID')
    tbheader['TARG_ID'] = (0, 'Target ID')
    tbheader['OBJECT'] = ('Fake X-1', 'Name of observed object')
    tbheader['RA_OBJ'] = (0.0, '[deg] R.A. Object')
    tbheader['DEC_OBJ'] = (0.0, '[deg] Dec Object')
    tbheader['RA_NOM'] = (0.0,
                          'Right Ascension used for barycenter corrections')
    tbheader['DEC_NOM'] = (0.0,
                           'Declination used for barycenter corrections')
    tbheader['RA_PNT'] = (0.0, '[deg] RA pointing')
    tbheader['DEC_PNT'] = (0.0, '[deg] Dec pointing')
    tbheader['PA_PNT'] = (0.0, '[deg] Position angle (roll)')
    tbheader['EQUINOX'] = (2.000E+03, 'Equinox of celestial coord system')
    tbheader['RADECSYS'] = ('FK5', 'Coordinate Reference System')
    tbheader['TASSIGN'] = ('SATELLITE', 'Time assigned by onboard clock')
    tbheader['TIMESYS'] = ('TDB', 'All times in this file are TDB')
    tbheader['MJDREFI'] = (int(mjdref),
                           'TDB time reference; Modified Julian Day (int)')
    tbheader['MJDREFF'] = (mjdref - int(mjdref),
                           'TDB time reference; Modified Julian Day (frac)')
    tbheader['TIMEREF'] = ('SOLARSYSTEM',
                           'Times are pathlength-corrected to barycenter')
    tbheader['CLOCKAPP'] = (False, 'TRUE if timestamps corrected by gnd sware')
    tbheader['COMMENT'] = ("MJDREFI+MJDREFF = epoch of Jan 1, 2010, in TT "
                           "time system.")
    tbheader['TIMEUNIT'] = ('s', 'unit for time keywords')
    tbheader['TSTART'] = (tstart,
                          'Elapsed seconds since MJDREF at start of file')
    tbheader['TSTOP'] = (tstop,
                         'Elapsed seconds since MJDREF at end of file')
    tbheader['LIVETIME'] = (livetime, 'On-source time')
    tbheader['TIMEZERO'] = (0.000000E+00, 'Time Zero')
    tbheader['COMMENT'] = (
        "Generated with MaLTPyNT by {0}".format(os.getenv('USER')))

    # ---- END Fake lots of information ----

    # Fake GTIs

    start = gti[:, 0]
    stop = gti[:, 1]

    col1 = fits.Column(name='START', format='1D', array=start)
    col2 = fits.Column(name='STOP', format='1D', array=stop)
    allcols = [col1, col2]
    cols = fits.ColDefs(allcols)
    gtihdu = fits.BinTableHDU.from_columns(cols)
    gtihdu.name = 'GTI'

    thdulist = fits.HDUList([prihdu, tbhdu, gtihdu])

    thdulist.writeto(filename, clobber=True)
    return thdulist


def _read_event_list(filename):
    if filename is not None:
        warnings.warn('Input event lists not yet implemented')
    return None, None


def _read_light_curve(filename):
    file_format = get_file_format(filename)
    if file_format == 'fits':
        filename = lcurve_from_fits(filename)[0]
    contents = load_lcurve(filename)

    return contents['time'], contents['lc']


def main(args=None):
    import argparse
    description = (
        'Create an event file in FITS format from an event list, or simulating'
        ' it. If input event list is not specified, generates the events '
        'randomly')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-e", "--event-list", type=str, default=None,
                        help="File containint event list")
    parser.add_argument("-l", "--lc", type=str, default=None,
                        help="File containing light curve")
    parser.add_argument("-c", "--ctrate", type=float, default=None,
                        help="Count rate for simulated events")
    parser.add_argument("-o", "--outname", type=str, default='events.evt',
                        help="Output file name")
    parser.add_argument("-i", "--instrument", type=str, default='FPMA',
                        help="Instrument name")
    parser.add_argument("--tstart", type=float, default=None,
                        help="Start time of the observation (s from MJDREF)")
    parser.add_argument("--tstop", type=float, default=None,
                        help="End time of the observation (s from MJDREF)")
    parser.add_argument("--mjdref", type=float, default=55197.00076601852,
                        help="Reference MJD")
    parser.add_argument("--deadtime", type=float, default=None, nargs='+',
                        help="Dead time magnitude. Can be specified as a "
                             "single number, or two. In this last case, the "
                             "second value is used as sigma of the dead time "
                             "distribution")

    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='MPfake.log', level=numeric_level,
                        filemode='w')

    additional_columns = {}
    livetime = None
    if args.lc is not None:
        t, lc = _read_light_curve(args.lc)
        event_list = fake_events_from_lc(t, lc, use_spline=True)
        pi = np.zeros(len(event_list), dtype=int)
        print('{} events generated'.format(len(event_list)))
    elif args.ctrate is not None:
        t = np.arange(0., 1025.)
        lc = args.ctrate + np.zeros_like(t)
        event_list = fake_events_from_lc(t, lc)
        pi = np.zeros(len(event_list), dtype=int)
        print('{} events generated'.format(len(event_list)))
    else:
        event_list, pi = _read_event_list(args.event_list)

    if args.deadtime is not None and event_list is not None:
        deadtime = args.deadtime[0]
        deadtime_sigma = None
        if len(args.deadtime) > 1:
            deadtime_sigma = args.deadtime[1]
        print(deadtime, deadtime_sigma)
        event_list, info = filter_for_deadtime(event_list, deadtime,
                                               dt_sigma=deadtime_sigma,
                                               return_all=True)
        print('{} events after filter'.format(len(event_list)))
        pi = pi[info.mask]
        prior = np.zeros_like(event_list)

        prior[1:] = np.diff(event_list) - info.deadtime[:-1]
        additional_columns["PRIOR"] = {"data": prior, "format": "D"}
        additional_columns["KIND"] = {"data": info.is_event, "format": "L"}
        livetime = np.sum(prior)

    generate_fake_fits_observation(event_list=event_list,
                                   filename=args.outname, pi=pi,
                                   instr='FPMA', tstart=args.tstart,
                                   tstop=args.tstop,
                                   mjdref=args.mjdref, livetime=livetime,
                                   additional_columns=additional_columns)
