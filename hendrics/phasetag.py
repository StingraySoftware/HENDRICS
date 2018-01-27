#!/usr/bin/env python
from __future__ import (division, print_function, unicode_literals,
                        absolute_import)

import argparse

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf

import warnings
from .io import high_precision_keyword_read, is_string
from .base import _assign_value_if_none
from .fold import fit_profile, std_fold_fit_func

from stingray.pulse.pulsar import pulse_phase, phase_exposure
#
#
# def parse_tim(timfile, format="tempo2"):
#     if format != "tempo2":
#         sys.exit("Only tempo2 files are understood for now")
#     file = open(timfile)
#     TOAs = []
#     for l in file.readlines():
#         els = l.split()
#         if len(els) < 5:
#             continue
#         dum, dum, TOA, dum, dum = els[:5]
#         TOAs.append(np.longdouble(TOA))
#     return np.array(TOAs)
#


def outfile_name(file):
    return file.replace(".evt", "_phasetag.evt")


def phase_tag_fits(filename, parameter_info, nbin=10,
                   ref_to_max=False, pepoch=None, expocorr=True,
                   pulse_ref_time=None, plot=True, test=False):
    """Phase-tag events in a FITS file with a given ephemeris.

    Parameters
    ----------
    filename : str
        Events FITS file
    parameter_info : str or array of floats
        If a string, this is a pulsar parameter file that PINT is able to
        understand. Otherwise, this is a list of frequency derivatives
        [F0, F1, F2, ...]

    Other parameters
    ----------------
    nbin : int
        Number of nbin in the pulsed profile
    ref_to_max : bool
        Automatically refer the TOAs to the maximum of the profile
    pepoch : float, default None
        Reference epoch for the timing solution. If None, this is the start
        of the observation.
    pulse_ref_time : float
        Reference time for the pulse. This overrides ref_to_max
    plot : bool
        Plot diagnostics
    expocorr : bool
        Use exposure correction when calculating the profile

    """

    outfile = outfile_name(filename)

    # ----- OPEN EVENT FITS FILE ------------------------

    hdulist = pf.open(filename, checksum=True)
    ref_mjd = high_precision_keyword_read(hdulist[1].header, 'MJDREF')
    timezero = high_precision_keyword_read(hdulist[1].header, 'TIMEZERO')

    hdulist.verify('warn')

    # ----- READ EVENT LIST ------------------------

    tbhdu = hdulist["EVENTS"]
    table = tbhdu.data

    ev_list = np.array(table.field("TIME"), dtype=np.longdouble)
    ev_list += timezero

    # ----- CHECK IF GTIs ARE PRESENT ------------------

    gtistring = None
    for gtiextname in ['STDGTI', 'GTI']:
        if not gtiextname in hdulist:
            continue
        gtihdu = hdulist[gtiextname]
        gtis = np.array(list(zip(gtihdu.data.field('START'),
                                 gtihdu.data.field('STOP'))))
        gtistring = gtiextname

    if gtistring is None:
        gtis = np.array([[ev_list[0], ev_list[-1]]])

    # ---- in MJD ----

    ev_mjd = ev_list / 86400 + ref_mjd
    gtis_mjd = gtis / 86400 + ref_mjd

    pepoch = _assign_value_if_none(pepoch, ev_mjd[0])

    # ------ Orbital DEMODULATION --------------------
    if is_string(parameter_info):
        times = -1
        frequency_derivatives = [0]
        raise NotImplementedError('This part is not yet implemented. Please '
                                  'use single frequencies and pepoch as '
                                  'documented')

    else:
        frequency_derivatives = parameter_info
        times = (ev_mjd - pepoch) * 86400

    f = frequency_derivatives[0]

    phase = pulse_phase(times, *frequency_derivatives, to_1=False)
    gti_phases = pulse_phase((gtis_mjd - pepoch) * 86400,
                             *frequency_derivatives, to_1=False)

    # ------- now apply period derivatives ------

    print("Calculating phases...", end='')
    ref_phase = 0
    ref_time = 0

    if ref_to_max:
        phase_to1 = phase - np.floor(phase)

        raw_profile, bins = np.histogram(phase_to1,
                                         bins=np.linspace(0, 1, nbin + 1))
        exposure = phase_exposure(gti_phases[0, 0], gti_phases[-1, 1], 1,
                                  nbin=nbin, gtis=gti_phases)
        profile = raw_profile / exposure
        profile_err = np.sqrt(raw_profile) / exposure

        sinpars, bu, bu = fit_profile(profile, profile_err,
                                      nperiods=2, baseline=True)
        fine_phases = np.linspace(0, 2, 1000 * 2)
        fitted_profile = std_fold_fit_func(sinpars, fine_phases)
        maxp = np.argmax(fitted_profile)
        ref_phase = fine_phases[maxp]
        if test:  # pragma: no cover
            ref_phase = bins[np.argmax(raw_profile)]

        ref_time = ref_phase / f
    elif pulse_ref_time is not None:
        raise NotImplementedError('pulse_ref_time is not implemented.')

    phase -= ref_phase
    gti_phases -= ref_phase
    phase_to1 = phase - np.floor(phase)

    raw_profile, bins = np.histogram(phase_to1,
                                     bins=np.linspace(0, 1, nbin + 1))

    exposure = phase_exposure(gti_phases[0, 0], gti_phases[-1, 1], 1,
                              nbin=nbin, gtis=gti_phases)
    if np.any(np.isnan(exposure)):
        warnings.warn('Exposure has NaNs. Profile is not normalized')
        expocorr = False

    if not expocorr:
        exposure = np.ones_like(raw_profile)

    profile = raw_profile / exposure
    profile = np.append(profile, profile)
    exposure = np.append(exposure, exposure)
    profile_err = np.sqrt(profile)
    phs = (bins[1:] + bins[:-1]) / 2
    phs = np.append(phs, phs + 1)

    if plot:
        plt.errorbar(phs, profile / exposure,
                     yerr=profile_err / exposure, fmt='none')
        plt.plot(phs, profile / exposure, 'k-',
                 drawstyle='steps-mid')
        plt.xlabel("Phase")
        plt.ylabel("Counts")
        for i in range(20):
            plt.axvline(i * 0.1, ls='--', color='b')
        if not test:  # pragma: no cover
            plt.show()
    # ------ WRITE RESULTS BACK TO FITS --------------

    # If columns already exist, overwrite them. Else, create them
    create = False
    if 'Orbit_bary' in table.names:
        table.field("Orbit_bary")[:] = ev_list
        table.field("TotPhase_s")[:] = phase / f + ref_time
        table.field("Phase")[:] = phase_to1
    else:
        create = True

    # first of all, copy columns
    cols = table.columns

    # create new list of columns, copying all columns from other table
    newlist = []

    for c in cols:
        newlist.append(c)

    if create:
        # then, create new column with orbital demodulation
        newcol = pf.Column(name='Orbit_bary',
                           format='1D',
                           unit='s',
                           array=ev_list)

        # append it to new table
        newlist.append(newcol)

        # Do the same with total phase
        newcol = pf.Column(name='TotPhase_s',
                           format='1D',
                           unit='s',
                           array=phase / f + ref_time)

        newlist.append(newcol)

        # Do the same with fractional phase
        newcol = pf.Column(name='Phase',
                           format='1D',
                           unit='phase',
                           array=phase_to1)

        newlist.append(newcol)

        # make new column definitions
    coldefs = pf.ColDefs(newlist)

    # create new record
    newrec = pf.FITS_rec.from_columns(coldefs)

    # and new hdu
    newtbhdu = pf.BinTableHDU(data=newrec,
                              header=tbhdu.header.copy(),
                              name='EVENTS', uint=False)

    # Copy primary HDU from old file
    prihdu = hdulist[0].copy()

    # define new hdulist
    newhdulist = pf.HDUList([prihdu, newtbhdu])

    # Copy remaining HDUs from old file
    if len(hdulist) > 2:
        for h in hdulist[2:]:
            newhdulist.append(h.copy())

    newhdulist.writeto(outfile, overwrite=True)
    hdulist.close()

#
# def phase_tag(filename, parfile, nperiods=2, nbin=10, ref_to_max=False,
#               pulse_ref_time=None, plot=True):
#
#     '''Phase-tags events according to ephemeris
#
#     filename is a event file in MaLTPyNT format
#     parfile is a parameter file in TEMPO2 format
#     '''
#     import libstempo.toasim as LT
#     import libstempo as T
#     from astropy.time import Time
#     from astropy import units as u
#     evdata = load_events(f)
#
#     pulsar = bp.binary_psr(parfile)
#
#     tstart = evdata['Tstart'] * u.s
#     tstop = evdata['Tstop'] * u.s
#     events = evdata['time'] * u.s
#     instr = evdata['Instr']
#     mjdref = Time(evdata['MJDref'], format='mjd')
#
#     if instr == 'PCA':
#         pcus = evdata['PCU']
#     gtis = evdata['GTI']
#     if ignore_gtis:
#         gtis = np.array([[tstart, tstop]])
#
#     ev_mjd = mjdref + events
#
#     newev = pulsar.demodulate_TOAs(ev_mjd.mjd)
#
#     newref_mjd, newpep_mjd = \
#         pulsar.demodulate_TOAs(np.array([mjdref.mjd, pulsar.par.PEPOCH]))


def main_phasetag(args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="Event file", type=str)
    parser.add_argument("--parfile", help="Parameter file", type=str,
                        default=None)
    parser.add_argument('-f', "--freqs", help="Frequency derivatives",
                        type=float, default=None, nargs='+')
    parser.add_argument("-n", "--nbin", type=int, default=16, help="Nbin")
    parser.add_argument("--plot", action="store_true", default=False,
                        dest="plot", help="Plot profile")
    parser.add_argument("--tomax", action="store_true", default=False,
                        dest="tomax",
                        help="Refer phase to pulse max")
    parser.add_argument("--test", action="store_true", default=False,
                        dest="test",
                        help="Only for unit tests! Do not use")
    parser.add_argument("--refTOA", default=None, type=float,
                        help="Reference TOA in MJD (overrides --tomax)")

    args = parser.parse_args(args)

    if args.freqs is None and args.parfile is None:
        raise ValueError('Specify one between --freqs or --parfile')
    elif args.freqs is not None and args.parfile is not None:
        raise ValueError('Specify only one between --freqs or --parfile')
    elif args.freqs is not None:
        parameter_info = args.freqs
    else:
        parameter_info = args.parfile

    plot = args.plot
    expocorr = True

    phase_tag_fits(args.file, parameter_info, plot=plot, nbin=args.nbin,
                   test=args.test, expocorr=expocorr, ref_to_max=args.tomax)
