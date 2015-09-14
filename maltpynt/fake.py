# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to simulate data and produce a fake event file.
"""
from __future__ import (absolute_import, division,
                        print_function)

import numpy as np
import numpy.random as ra
import os
import logging
import warnings
from .io import get_file_format, load_lcurve
from .lcurve import lcurve_from_fits


def fake_events_from_lc(
        times, lc, use_spline=False, bin_time=None):  # pragma: no cover
    '''
    Create events from a light curve.

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
    '''
    try:
        import scipy.interpolate as sci
    except:
        if use_spline:
            warnings.warn("Scipy not available. "
                          "use_spline option cannot be used.")
            use_spline = False

    if bin_time is None:
        bin_time = times[1] - times[0]
    n_bin = len(lc)

    bin_start = 0

    n_events_predict = np.max(lc) * n_bin
    n_events_predict += 10 * np.sqrt(n_events_predict)

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
        logging.debug(t_filt[0] - bin_time / 2,
                      t_filt[-1] + bin_time / 2)

        length = t_filt[-1] - t_filt[0]
        n_bin_filt = len(lc_filt)
        n_to_simulate = n_bin_filt * max(lc_filt)
        safety_factor = 10
        if n_to_simulate > 1000:
            safety_factor = 1.

        n_to_simulate += safety_factor * np.sqrt(n_to_simulate)

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
    return ev_list[:nev]


def filter_for_deadtime(ev_list, deadtime, bkg_ev_list=None,
                        dt_sigma=None, paralyzable=False):
    '''
    Filter an event list for a given dead time.
    Parameters
    ----------
    ev_list : array-like
        The event list
    deadtime: float
        The (mean, if not constant) value of deadtime
    bkg_ev_list : array-like, optional
        A background event list that affects dead time
    dt_sigma : float, optional
        The standard deviation of a non-constant dead time around deadtime.

    Returns
    -------
    new_ev_list : array-like
        The filtered event list
    new_bkg_ev : array-like
        The filtered background event list. Only returned if background_ev_list
        is not None)

    '''
    if deadtime <= 0.:
        return np.copy(ev_list)

    # Create the total lightcurve, and a "kind" array that keeps track
    # of the events classified as "signal" (True) and "background" (False)
    if bkg_ev_list is not None:
        return_bkg = True
        tot_ev_list = np.append(ev_list, bkg_ev_list)
        ev_kind = np.append(np.ones(len(ev_list), dtype=bool),
                            np.zeros(len(bkg_ev_list), dtype=bool))
        order = np.argsort(tot_ev_list)
        tot_ev_list = tot_ev_list[order]
        ev_kind = ev_kind[order]
        del order
    else:
        return_bkg = False
        tot_ev_list = ev_list
        ev_kind = np.ones(len(ev_list), dtype=bool)

    mask = np.ones(len(tot_ev_list), dtype=bool)
    nevents = len(tot_ev_list)

    if dt_sigma is not None:
        deadtime_values = ra.normal(deadtime, dt_sigma, nevents)
    else:
        deadtime_values = np.zeros(nevents) + deadtime

    dead_time_end = tot_ev_list + deadtime_values

    if paralyzable:
        bad = dead_time_end[:-1] > tot_ev_list[1:]
        # Easy: paralyzable case. Here, events coming during dead time produce
        # more dead time. So...
        mask[1:][bad] = False
        tot_ev_list = tot_ev_list[mask]
        ev_kind = ev_kind[mask]
    else:
        # Otherwise, it is a little trickier. An event is filtered if it comes
        # during dead time AND the previous event was valid. We need to iterate
        while True:
            mask_2 = np.zeros_like(mask)

            before_deadtime = dead_time_end[:-1] > tot_ev_list[1:]
            mask_2[1:] = before_deadtime
            bad = np.logical_and(mask_2[1:] == True,
                                 mask_2[:-1] == 0)

            mask[1:] = np.logical_not(bad)

            if np.all(mask):
                break

            tot_ev_list = tot_ev_list[mask]
            ev_kind = ev_kind[mask]
            deadtime_values = deadtime_values[mask]
            dead_time_end = tot_ev_list + deadtime_values
            mask = mask[mask]

    if return_bkg:
        return tot_ev_list[ev_kind], tot_ev_list[np.logical_not(ev_kind)]
    else:
        return tot_ev_list[ev_kind]


def generate_fake_fits_observation(event_list=None, filename=None, pi=None,
                                   instr='FPMA', gti=None, tstart=None,
                                   tstop=None,
                                   mjdref=55197.00076601852,
                                   livetime=None):
    '''Generate fake NuSTAR data!

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
    '''
    from astropy.io import fits
    import numpy.random as ra

    if event_list is None:
        if tstart is None:
            tstart = 8e+7
        if tstop is None:
            tstop = tstart + 1025
        event_list = sorted(ra.uniform(tstart, tstop, 1000))

    if pi is None:
        pi = ra.randint(0, 1024, len(event_list))

    assert len(event_list) == len(pi), \
        "Event list and pi must be of the same length"

    if tstart is None:
        tstart = np.floor(event_list[0])

    if tstop is None:
        tstop = np.ceil(event_list[-1])

    if gti is None:
        gti = np.array([[tstart, tstop]])

    if filename is None:
        filename = 'events.evt'

    if livetime is None:
        livetime = tstop - tstart

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
    cols = fits.ColDefs([col1, col2])

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
    cols = fits.ColDefs([col1, col2])
    gtihdu = fits.BinTableHDU.from_columns(cols)
    gtihdu.name = 'GTI'

    thdulist = fits.HDUList([prihdu, tbhdu, gtihdu])

    thdulist.writeto(filename, clobber=True)
    return thdulist


def _read_event_list(filename):
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
    parser.add_argument("--deadtime", type=float, default=0.,
                        help="Reference MJD")

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

    if args.lc is not None:
        t, lc = _read_light_curve(args.lc)
        event_list = fake_events_from_lc(t, lc, use_spline=True)
        pi = np.zeros(len(event_list), dtype=int)
    else:
        event_list, pi = _read_event_list(args.event_list)

    event_list = filter_for_deadtime(event_list, args.deadtime)

    generate_fake_fits_observation(event_list=event_list,
                                   filename=args.outname, pi=pi,
                                   instr='FPMA', tstart=args.tstart,
                                   tstop=args.tstop,
                                   mjdref=args.mjdref)
