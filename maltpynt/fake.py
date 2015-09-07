# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate the exposure correction for light curves. Only works for data
   taken in specific data modes of NuSTAR, where all events are telemetered.

"""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
import os
import logging


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
        event_list = ra.uniform(tstart, tstop, 1000)

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


def main(args=None):
    import argparse
    description = (
        'Create an event file in FITS format from an event list, or simulating'
        ' it. If input event list is not specified, generates the events '
        'randomly')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-e", "--event-list", type=str, default=None,
                        help="File containint event list")
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

    event_list, pi = _read_event_list(args.event_list)
    generate_fake_fits_observation(event_list=event_list,
                                   filename=args.outname, pi=pi,
                                   instr='FPMA', tstart=args.tstart,
                                   tstop=args.tstop,
                                   mjdref=args.mjdref)
