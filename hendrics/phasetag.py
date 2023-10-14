#!/usr/bin/env python

import os
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf
from astropy import log
from astropy.logger import AstropyUserWarning

from stingray.io import load_events_and_gtis, ref_mjd
from stingray.pulse.pulsar import pulse_phase, phase_exposure
from .io import is_string, save_as_qdp
from .base import _assign_value_if_none, hen_root, splitext_improved
from .fold import fit_profile, std_fold_fit_func


def outfile_name(file):
    """Output file name for phasetag.

    Examples
    --------
    >>> outfile_name('file.s.a.fits.Z')
    'file.s.a_phasetag.fits.Z'
    >>> outfile_name('file.s.a.evct.gz')
    'file.s.a_phasetag.evct.gz'
    >>> outfile_name('file.s.a.evct')
    'file.s.a_phasetag.evct'
    """

    root, ext = splitext_improved(file)

    return root + "_phasetag" + ext


def phase_tag(
    ev_list,
    parameter_info,
    gti=None,
    mjdref=0,
    nbin=10,
    ref_to_max=False,
    pepoch=None,
    expocorr=True,
    pulse_ref_time=None,
    plot=True,
    test=False,
):
    """Phase-tag events in a FITS file with a given ephemeris.

    Parameters
    ----------
    ev_list : float
        Event times
    parameter_info : str or array of floats
        If a string, this is a pulsar parameter file that PINT is able to
        understand. Otherwise, this is a list of frequency derivatives
        [F0, F1, F2, ...]

    Other parameters
    ----------------
    gti : [[g0_0, g0_1], [g1_0, g1_1], ...]
        Good time intervals
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
    # ---- in MJD ----
    if gti is None:
        gti = np.array([[ev_list[0], ev_list[-1]]])

    ev_mjd = ev_list / 86400 + mjdref
    gti_mjd = gti / 86400 + mjdref

    pepoch = _assign_value_if_none(pepoch, gti_mjd[0, 0])

    # ------ Orbital DEMODULATION --------------------
    if is_string(parameter_info):
        raise NotImplementedError(
            "This part is not yet implemented. Please "
            "use single frequencies and pepoch as "
            "documented"
        )

    else:
        frequency_derivatives = parameter_info
        times = (ev_mjd - pepoch) * 86400

    f = frequency_derivatives[0]

    phase = pulse_phase(times, *frequency_derivatives, to_1=False)
    gti_phases = pulse_phase(
        (gti_mjd - pepoch) * 86400, *frequency_derivatives, to_1=False
    )

    # ------- now apply period derivatives ------

    log.info("Calculating phases...")
    ref_phase = 0
    ref_time = 0

    if pulse_ref_time is not None:
        ref_time = (pulse_ref_time - pepoch) * 86400
        ref_phase = ref_time * f
    elif ref_to_max:
        phase_to1 = phase - np.floor(phase)

        raw_profile, bins = np.histogram(phase_to1, bins=np.linspace(0, 1, nbin + 1))
        exposure = phase_exposure(
            gti_phases[0, 0], gti_phases[-1, 1], 1, nbin=nbin, gti=gti_phases
        )
        profile = raw_profile / exposure
        profile_err = np.sqrt(raw_profile) / exposure

        sinpars, bu, bu = fit_profile(
            profile, profile_err, nperiods=2, baseline=True, debug=test
        )
        fine_phases = np.linspace(0, 2, 1000 * 2)
        fitted_profile = std_fold_fit_func(sinpars, fine_phases)
        maxp = np.argmax(fitted_profile)
        ref_phase = fine_phases[maxp]
        if test:  # pragma: no cover
            # No tests with a pulsed profile yet
            ref_phase = bins[np.argmax(raw_profile)]

        ref_time = ref_phase / f

    phase -= ref_phase
    gti_phases -= ref_phase
    phase_to1 = phase - np.floor(phase)

    raw_profile, bins = np.histogram(phase_to1, bins=np.linspace(0, 1, nbin + 1))

    exposure = phase_exposure(
        gti_phases[0, 0], gti_phases[-1, 1], 1, nbin=nbin, gti=gti_phases
    )
    if np.any(np.logical_or(exposure != exposure, exposure == 0)):
        warnings.warn(
            "Exposure has NaNs or zeros. Profile is not normalized",
            AstropyUserWarning,
        )
        expocorr = False

    if not expocorr:
        exposure = np.ones_like(raw_profile)

    profile = raw_profile / exposure
    profile = np.append(profile, profile)
    exposure = np.append(exposure, exposure)
    profile_err = np.sqrt(profile)
    phs = (bins[1:] + bins[:-1]) / 2
    phs = np.append(phs, phs + 1)

    fig = None
    if plot:
        fig = plt.figure()
        plt.errorbar(phs, profile / exposure, yerr=profile_err / exposure, fmt="none")
        plt.plot(phs, profile / exposure, "k-", drawstyle="steps-mid")
        plt.xlabel("Phase")
        plt.ylabel("Counts")
        for i in range(20):
            plt.axvline(i * 0.1, ls="--", color="b")
        if not test:  # pragma: no cover
            plt.show()
        else:
            plt.close(fig)

    # ------ WRITE RESULTS BACK TO FITS --------------
    results = type("results", (object,), {})
    results.ev_list = ev_list
    results.phase = phase
    results.frequency_derivatives = frequency_derivatives
    results.ref_time = ref_time
    results.figure = fig
    results.plot_phase = phs
    results.plot_profile = profile / exposure
    results.plot_profile_err = profile_err / exposure
    return results


def phase_tag_fits(
    filename,
    parameter_info,
    gtistring="GTI,STDGTI",
    gti_file=None,
    hduname="EVENTS",
    column="TIME",
    **kwargs
):
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
    gtistring : str
        Comma-separated list of accepted GTI extensions (default GTI,STDGTI),
        with or without appended integer number denoting the detector
    gti_file : str, default None
        External GTI file
    hduname : str, default 'EVENTS'
        Name of the HDU containing the event list
    """

    outfile = outfile_name(filename)
    evreturns = load_events_and_gtis(
        filename,
        gtistring=gtistring,
        gti_file=gti_file,
        hduname=hduname,
        column=column,
    )
    mjdref = ref_mjd(filename)

    results = phase_tag(
        evreturns.ev_list,
        parameter_info,
        gti=evreturns.gti_list,
        mjdref=mjdref,
        **kwargs
    )
    if results.figure is not None:
        results.figure.savefig(hen_root(filename) + ".pdf")
    phase = results.phase
    frequency_derivatives = results.frequency_derivatives
    ref_time = results.ref_time

    phase_to1 = phase - np.floor(phase)

    # Save results to fits file
    hdulist = pf.open(filename, checksum=True)
    tbhdu = hdulist[hduname]
    table = tbhdu.data
    order = np.argsort(table[column])
    # load_events_and_gtis sorts the data automatically. This is not the case
    # of the other operations done here. So, let's sort the table first.
    for col in table.names:
        table.field(col)[:] = np.array(table[col])[order]

    # If columns already exist, overwrite them. Else, create them
    create = False
    if "Orbit_bary" in table.names:
        table.field("Orbit_bary")[:] = evreturns.ev_list
        table.field("TotPhase_s")[:] = phase / frequency_derivatives[0] + ref_time
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
        newcol = pf.Column(
            name="Orbit_bary", format="1D", unit="s", array=results.ev_list
        )

        # append it to new table
        newlist.append(newcol)

        # Do the same with total phase
        newcol = pf.Column(
            name="TotPhase_s",
            format="1D",
            unit="s",
            array=phase / frequency_derivatives[0] + ref_time,
        )

        newlist.append(newcol)

        # Do the same with fractional phase
        newcol = pf.Column(name="Phase", format="1D", unit="phase", array=phase_to1)

        newlist.append(newcol)

        # make new column definitions
    coldefs = pf.ColDefs(newlist)

    # create new record
    newrec = pf.FITS_rec.from_columns(coldefs)

    # and new hdu
    newtbhdu = pf.BinTableHDU(
        data=newrec, header=tbhdu.header.copy(), name=hduname, uint=False
    )

    # Copy primary HDU from old file
    prihdu = hdulist[0].copy()

    # define new hdulist
    newhdulist = pf.HDUList([prihdu])

    # Copy remaining HDUs from old file
    for h in hdulist[1:]:
        if h.name == hduname:
            newhdulist.append(newtbhdu)
        else:
            newhdulist.append(h.copy())

    try:
        newhdulist.writeto(outfile, overwrite=True, checksum=True)
    except Exception:
        newhdulist.writeto(outfile, overwrite=True)
    hdulist.close()

    save_as_qdp(
        [results.plot_phase, results.plot_profile, results.plot_profile_err],
        filename=outfile.replace(".evt", "") + ".qdp",
    )


def main_phasetag(args=None):
    from .base import check_negative_numbers_in_args

    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="Event file", type=str)
    parser.add_argument("--parfile", help="Parameter file", type=str, default=None)
    parser.add_argument(
        "-f",
        "--freqs",
        help="Frequency derivatives",
        type=float,
        default=None,
        nargs="+",
    )
    parser.add_argument("-n", "--nbin", type=int, default=16, help="Nbin")
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        dest="plot",
        help="Plot profile",
    )
    parser.add_argument(
        "--tomax",
        action="store_true",
        default=False,
        dest="tomax",
        help="Refer phase to pulse max",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        dest="test",
        help="Only for unit tests! Do not use",
    )
    parser.add_argument(
        "--refTOA",
        default=None,
        type=np.longdouble,
        help="Reference TOA in MJD (overrides --tomax) for " "reference pulse phase",
        dest="pulse_ref_time",
    )
    parser.add_argument(
        "--pepoch",
        default=None,
        type=np.longdouble,
        help="Reference time for timing solution",
        dest="pepoch",
    )

    args = check_negative_numbers_in_args(args)
    args = parser.parse_args(args)

    if args.freqs is None and args.parfile is None:
        raise ValueError("Specify one between --freqs or --parfile")
    elif args.freqs is not None and args.parfile is not None:
        raise ValueError("Specify only one between --freqs or --parfile")
    elif args.freqs is not None:
        parameter_info = args.freqs
    else:
        parameter_info = args.parfile

    plot = args.plot
    expocorr = True

    phase_tag_fits(
        args.file,
        parameter_info,
        plot=plot,
        nbin=args.nbin,
        test=args.test,
        expocorr=expocorr,
        ref_to_max=args.tomax,
        pulse_ref_time=args.pulse_ref_time,
        pepoch=args.pepoch,
    )
