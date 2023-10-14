# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to simulate data and produce a fake event file."""

import os
import warnings
import copy
import numpy as np
import numpy.random as ra
from astropy import log
from astropy.io.fits import Header
from astropy.logger import AstropyUserWarning
from stingray.events import EventList
from stingray.lightcurve import Lightcurve
from stingray.utils import assign_value_if_none
from stingray.filters import filter_for_deadtime
from stingray.io import read_mission_info, get_key_from_mission_info
from .io import load_lcurve
from .io import load_events, save_events, HEN_FILE_EXTENSION
from .base import _empty, get_file_format, r_in
from .fold import filter_energy
from .lcurve import lcurve_from_fits
from .base import njit, deorbit_events


def _clean_up_header(header):
    if header is None:
        return None
    for key in header.keys():
        for k in ["TTYP", "TFORM"]:
            if key.startswith(k):
                header.pop(key)
    for k in ["EXTNAME", "PCOUNT", "GCOUNT", "NAXIS1", "NAXIS2"]:
        if k in header:
            header.pop(k)
    return header


def _fill_in_default_information(tbheader):
    tbheader["OBSERVER"] = "Edwige Bubble"
    tbheader["COMMENT"] = (
        "FITS (Flexible Image Transport System) format is"
        " defined in 'Astronomy and Astrophysics', volume"
        " 376, page 359; bibcode: 2001A&A...376..359H"
    )
    tbheader["OBS_ID"] = ("00000000001", "Observation ID")
    tbheader["TARG_ID"] = (0, "Target ID")
    tbheader["OBJECT"] = ("Fake X-1", "Name of observed object")
    tbheader["RA_OBJ"] = (0.0, "[deg] R.A. Object")
    tbheader["DEC_OBJ"] = (0.0, "[deg] Dec Object")
    tbheader["RA_NOM"] = (
        0.0,
        "Right Ascension used for barycenter corrections",
    )
    tbheader["DEC_NOM"] = (0.0, "Declination used for barycenter corrections")
    tbheader["RA_PNT"] = (0.0, "[deg] RA pointing")
    tbheader["DEC_PNT"] = (0.0, "[deg] Dec pointing")
    tbheader["PA_PNT"] = (0.0, "[deg] Position angle (roll)")
    tbheader["EQUINOX"] = (2.000e03, "Equinox of celestial coord system")
    tbheader["RADECSYS"] = ("FK5", "Coordinate Reference System")
    tbheader["TASSIGN"] = ("SATELLITE", "Time assigned by onboard clock")
    tbheader["TIMESYS"] = ("TDB", "All times in this file are TDB")
    tbheader["TIMEREF"] = (
        "SOLARSYSTEM",
        "Times are pathlength-corrected to barycenter",
    )
    tbheader["CLOCKAPP"] = (
        False,
        "TRUE if timestamps corrected by gnd sware",
    )
    tbheader["COMMENT"] = (
        "MJDREFI+MJDREFF = epoch of Jan 1, 2010, in TT " "time system."
    )
    tbheader["TIMEUNIT"] = ("s", "unit for time keywords")
    return tbheader


def generate_fake_fits_observation(
    event_list=None,
    filename=None,
    instr=None,
    gti=None,
    tstart=None,
    tstop=None,
    mission=None,
    mjdref=55197.00076601852,
    livetime=None,
    additional_columns={},
):
    """Generate fake X-ray data.

    Takes an event list (as a list of floats)
    All inputs are None by default, and can be set during the call.

    Parameters
    ----------
    event_list : list-like
        :class:`stingray.events.Eventlist` object. If left None, 1000
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

    inheader = None
    if event_list is None:
        tstart = assign_value_if_none(tstart, 8e7)
        tstop = assign_value_if_none(tstop, tstart + 1025)
        ev_list = np.sort(ra.uniform(tstart, tstop, 1000))
        gti = assign_value_if_none(gti, np.array([[tstart, tstop]]))
    else:
        if hasattr(event_list, "header") and event_list.header is not None:
            inheader = Header.fromstring(event_list.header)
            inheader = _clean_up_header(inheader)

        ev_list = event_list.time
        gti = assign_value_if_none(
            event_list.gti, np.asarray([[ev_list[0], ev_list[-1]]])
        )
        mission = assign_value_if_none(mission, event_list.mission)
        instr = assign_value_if_none(instr, event_list.instr)
        tstart = assign_value_if_none(tstart, gti[0, 0])
        tstop = assign_value_if_none(tstop, gti[-1, 1])
        if hasattr(event_list, "mjdref") and event_list.mjdref is not None:
            mjdref = event_list.mjdref

    mission = assign_value_if_none(mission, "NuSTAR")
    instr = assign_value_if_none(instr, "FPMA")
    if hasattr(event_list, "pi") and event_list.pi is not None:
        pi = event_list.pi
    else:
        pi = ra.randint(0, 1024, np.size(ev_list))

    if hasattr(event_list, "cal_pi") and event_list.cal_pi is not None:
        cal_pi = event_list.cal_pi
    else:
        cal_pi = pi / 3

    filename = assign_value_if_none(filename, "events.evt")
    livetime = assign_value_if_none(livetime, tstop - tstart)

    if livetime > tstop - tstart:
        raise ValueError("Livetime must be equal or smaller than " "tstop - tstart")

    mission_info = read_mission_info(mission)
    allowed_instr = []
    if "instruments" in mission_info:
        allowed_instr = mission_info["instruments"]

    # Just prefer EPN for XMM
    if "xmm" in mission.lower():
        allowed_instr = ["EPN", "EMOS1", "EMOS2", "RGS1", "RGS2"]

    allowed_instr = [ins.lower() for ins in allowed_instr]
    if (allowed_instr != []) and (instr.lower() not in allowed_instr):
        instr = allowed_instr[0]

    ccol = get_key_from_mission_info(mission_info, "ccol", "none", inst=instr)
    if ccol is not None and ccol.lower() == "none":
        ccol = None
    ecol = get_key_from_mission_info(mission_info, "ecol", "PI", inst=instr)
    ext = get_key_from_mission_info(mission_info, "events", "EVENTS", inst=instr)

    # Create primary header
    prihdr = fits.Header()
    prihdr["OBSERVER"] = "Edwige Bubble"
    if inheader is not None and "OBSERVER" in inheader:
        prihdr["OBSERVER"] = inheader["OBSERVER"]
    prihdr["TELESCOP"] = (mission, "Telescope (mission) name")
    prihdr["INSTRUME"] = (instr, "Instrument name")
    prihdu = fits.PrimaryHDU(header=prihdr)
    prihdu.verify("exception")

    # Write events to table
    col1 = fits.Column(name="TIME", format="1D", array=ev_list)

    allcols = [col1]

    if ccol is not None:
        if not hasattr(event_list, "detector_id") or event_list.detector_id is None:
            ccdnr = np.zeros(np.size(ev_list)) + 1
            ccdnr[1] = 2  # Make it less trivial
            ccdnr[10] = 7
        else:
            ccdnr = event_list.detector_id

        allcols.append(fits.Column(name=ccol, format="1J", array=ccdnr))

    if mission.lower().strip() in ["xmm", "swift"]:
        allcols.append(fits.Column(name="PHA", format="1J", array=pi))
        allcols.append(fits.Column(name="PI", format="1J", array=cal_pi))
    else:
        allcols.append(fits.Column(name=ecol, format="1J", array=pi))

    for c in additional_columns.keys():
        col = fits.Column(
            name=c,
            array=additional_columns[c]["data"],
            format=additional_columns[c]["format"],
        )
        allcols.append(col)

    cols = fits.ColDefs(allcols)

    # ---- Fake lots of information ----

    tbheader = Header()
    tbheader = _fill_in_default_information(tbheader)
    # If None, it will not update
    tbheader.update(inheader)

    tbheader["TSTART"] = (
        tstart,
        "Elapsed seconds since MJDREF at start of file",
    )
    tbheader["TELESCOP"] = (mission, "Telescope (mission) name")
    tbheader["INSTRUME"] = (instr, "Instrument name")
    if mjdref != float(int(mjdref)):
        tbheader["MJDREFI"] = (
            int(mjdref),
            "TDB time reference; Modified Julian Day (int)",
        )
        tbheader["MJDREFF"] = (
            mjdref - int(mjdref),
            "TDB time reference; Modified Julian Day (frac)",
        )
        tbheader.pop("MJDREF", None)
    else:
        tbheader["MJDREF"] = mjdref
        tbheader.pop("MJDREFI", None)
        tbheader.pop("MJDREFF", None)

    tbheader["TSTOP"] = (tstop, "Elapsed seconds since MJDREF at end of file")
    tbheader["LIVETIME"] = (livetime, "On-source time")
    tbheader["TIMEZERO"] = (0.000000e00, "Time Zero")
    tbheader["HISTORY"] = "Generated with HENDRICS by {0}".format(os.getenv("USER"))

    tbhdu = fits.BinTableHDU.from_columns(cols, header=tbheader)
    tbhdu.name = ext

    tbhdu.add_checksum()
    tbhdu.verify("exception")
    # ---- END Fake lots of information ----

    # Fake GTIs

    start = gti[:, 0]
    stop = gti[:, 1]

    col1 = fits.Column(name="START", format="1D", array=start)
    col2 = fits.Column(name="STOP", format="1D", array=stop)
    allcols = [col1, col2]
    cols = fits.ColDefs(allcols)
    gtinames = ["GTI"]
    if mission.lower().strip() == "xmm":
        gtinames = []
        for i in set(ccdnr):
            gtinames.append(f"STDGTI{int(i):02d}")

    all_new_hdus = [prihdu, tbhdu]
    for name in gtinames:
        gtihdu = fits.BinTableHDU.from_columns(cols)
        gtihdu.name = name
        gtihdu.verify("exception")
        all_new_hdus.append(gtihdu)

    tbhdu.verify("exception")

    thdulist = fits.HDUList(all_new_hdus)
    assert thdulist[1].verify_datasum() == 1
    thdulist.writeto(filename, overwrite=True, checksum=True, output_verify="exception")

    thdulist.close()
    return filename


def _read_event_list(filename):
    ev_list = load_events(filename)
    return ev_list


def _read_light_curve(filename):
    file_format = get_file_format(filename)
    if file_format == "fits":
        filename = lcurve_from_fits(filename)[0]
    lc = load_lcurve(filename)

    return lc


def acceptance_rejection(
    dt, counts_per_bin, t0=0.0, poissonize_n_events=False, deadtime=0.0
):
    """
    Examples
    --------
    >>> counts_per_bin = [10, 5, 5]
    >>> dt = 0.1
    >>> ev = acceptance_rejection(dt, counts_per_bin)
    >>> ev.size == 20
    True
    >>> ev.max() < 0.3
    True
    >>> ev.min() > 0
    True
    >>> np.all(np.diff(ev) >= 0)
    True
    """
    counts_per_bin = np.asarray(counts_per_bin)
    rates = counts_per_bin / dt
    dead_time_corrected_rates = r_in(deadtime, rates)
    counts_per_bin = dead_time_corrected_rates * dt

    n_events = np.rint(np.sum(counts_per_bin)).astype(int)
    if poissonize_n_events:
        n_events = np.random.poisson(n_events)

    n_bins = counts_per_bin.size
    event_times = np.zeros(n_events)
    n_missing = n_events
    M = np.max(counts_per_bin)

    while n_missing > 0:
        stats = np.random.uniform(0, M, n_missing)
        float_bin = np.random.uniform(0, n_bins, n_missing)

        int_bin = np.floor(float_bin).astype(int)
        good = stats < counts_per_bin[int_bin]
        n = np.count_nonzero(good)
        if n == 0:
            continue
        start_bin = -n_missing
        end_bin = -n_missing + n
        if end_bin == 0:
            end_bin = event_times.size

        event_times[start_bin:end_bin] = float_bin[good] * dt + t0
        n_missing -= n

    return filter_for_deadtime(np.sort(event_times), deadtime)


def make_counts_pulsed(nevents, t_start, t_stop, pulsed_fraction=0.0):
    """

    Examples
    --------
    >>> nevents = 10
    >>> dt, counts = make_counts_pulsed(nevents, 0, 100)
    >>> np.isclose(np.sum(counts), nevents)
    True
    >>> dt, counts = make_counts_pulsed(nevents, 0, 100, pulsed_fraction=1)
    >>> np.isclose(np.sum(counts), nevents)
    True
    """
    dt = 0.0546372810934756
    length = t_start - t_stop
    n_bins = int(np.ceil(length / dt))
    # make dt an exact divisor of the length
    dt = length / n_bins

    times = np.arange(t_start, t_stop, dt)
    sinusoid = pulsed_fraction / 2 * np.sin(np.pi * 2 * times)

    lc = 1 - pulsed_fraction / 2 + sinusoid

    counts = lc * nevents / np.sum(lc)
    return dt, counts


def scramble(
    event_list,
    smooth_kind="flat",
    dt=None,
    pulsed_fraction=0.0,
    deadtime=0.0,
    orbit_par=None,
    frequency=1,
):
    """Scramble event list, GTI by GTI.

    Parameters
    ----------
    event_list: :class:`stingray.events.Eventlist` object
        Input event list

    Other parameters
    ----------------
    smooth_kind: str in ['flat', 'smooth', 'pulsed']
        if 'flat', count the events GTI by GTI without caring about long-term
        variability; if 'smooth', try to calculate smooth light curve first
    dt: float
        If ``smooth_kind`` is 'smooth', bin the light curve with this bin time.
        Ignored for other values of ``smooth_kind``
    pulsed_fraction: float
        If ``smooth_kind`` is 'pulsed', use this pulse fraction, defined as the
        2 A / B, where A is the amplitude of the sinusoid and B the maximum
        flux. Ignored for other values of ``smooth_kind``
    deadtime: float
        Dead time in the data.
    orbit_par: str
        Parameter file for orbital modulation

    Returns
    -------
    new_event_list: :class:`stingray.events.Eventlist` object
        "Scrambled" event list

    Examples
    --------
    >>> times = np.array([0.5, 134, 246, 344, 867])
    >>> event_list = EventList(
    ...     times, gti=np.array([[0, 0.9], [111, 123.2], [125.123, 1000]]))
    >>> new_event_list = scramble(event_list, 'smooth')
    >>> new_event_list.time.size == times.size
    True
    >>> np.all(new_event_list.gti == event_list.gti)
    True
    >>> new_event_list = scramble(event_list, 'flat')
    >>> new_event_list.time.size == times.size
    True
    >>> np.all(new_event_list.gti == event_list.gti)
    True
    """
    new_event_list = copy.deepcopy(event_list)
    assert np.all(np.diff(new_event_list.time) > 0)

    idxs = np.searchsorted(new_event_list.time, new_event_list.gti)

    if smooth_kind == "pulsed":
        # Frequency is one, but can be anywhere in the frequency bin (for
        # sensitivity losses)
        length = new_event_list.gti.max() - new_event_list.gti.min()
        df = 0.5 / length
        frequency += np.random.uniform(-df, df)

    for (i_start, i_stop), gti_boundary in zip(idxs, new_event_list.gti):
        locally_flat = False
        nevents = i_stop - i_start
        t_start, t_stop = gti_boundary[0], gti_boundary[1]
        if nevents < 1:
            continue
        length = t_stop - t_start
        if nevents < 10 and smooth_kind == "pulsed":
            continue
        if length <= 1:
            # in very short GTIs, always assume a flat distribution.
            locally_flat = True

        if smooth_kind == "flat" or locally_flat:
            rate = nevents / length
            input_rate = r_in(deadtime, rate)
            new_events = np.sort(
                np.random.uniform(
                    t_start,
                    t_stop,
                    np.rint(nevents * input_rate / rate).astype(int),
                )
            )
            new_events = filter_for_deadtime(new_events, deadtime)
            new_event_list.time[i_start:i_stop] = new_events[: i_stop - i_start]
            continue
        elif smooth_kind == "smooth":
            if dt is None:
                # Try to have at least 20 counts per bin on average
                dt = min(length / (nevents / 20), length)
            # make dt an exact divisor of the length
            n_bins = int(np.ceil(length / dt))
            dt = length / n_bins

            counts, _ = np.histogram(
                new_event_list.time[i_start:i_stop],
                range=[t_start, t_stop],
                bins=n_bins,
            )

        elif smooth_kind == "pulsed":
            # dt must be sufficiently small so that the frequency can be
            # detected with no loss of sensitivity. Moreover, not exactly a
            # multiple of the frequency, to increase randomness in the
            # detection sensitivity. I take a random Nyquist frequency between
            # 10 and 15 times the pulse frequency
            nyq = np.random.uniform(frequency * 10, frequency * 15)
            dt = 0.5 / nyq
            n_bins = int(np.ceil(length / dt))
            # make dt an exact divisor of the length
            dt = length / n_bins

            times = np.arange(t_start, t_stop, dt)
            sinusoid = pulsed_fraction / 2 * np.sin(np.pi * 2 * times * frequency)

            lc = 1 - pulsed_fraction / 2 + sinusoid

            counts = lc * nevents / np.sum(lc)
        else:
            raise ValueError("Unknown value for `smooth_kind`")

        newev = acceptance_rejection(
            dt,
            counts,
            t0=t_start,
            poissonize_n_events=False,
            deadtime=deadtime,
        )
        new_event_list.time[i_start:i_stop] = newev

    if orbit_par is not None:
        new_event_list = deorbit_events(new_event_list, orbit_par, invert=True)

    return new_event_list


def main_scramble(args=None):
    """Main function called by the `HENscramble` command line script."""
    import argparse
    from .base import _add_default_args, check_negative_numbers_in_args

    description = (
        "Scramble the events inside an event list, maintaining the same "
        "energies and GTIs"
    )
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "fname",
        type=str,
        default=None,
        help="File containing input event list",
    )
    parser.add_argument(
        "--smooth-kind",
        choices=["smooth", "flat", "pulsed"],
        help="Special testing value",
        default="flat",
    )
    parser.add_argument(
        "--deadtime",
        type=float,
        default=0,
        help="Dead time magnitude. Can be specified as a "
        "single number, or two. In this last case, the "
        "second value is used as sigma of the dead time "
        "distribution",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0,
        help="Time resolution of smoothed light curve",
    )
    parser.add_argument(
        "--pulsed-fraction",
        type=float,
        default=0,
        help="Pulsed fraction of simulated pulsations",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        type=float,
        default=1,
        help="Pulsed fraction of simulated pulsations",
    )
    parser.add_argument("--outfile", type=str, default=None, help="Output file name")
    args = check_negative_numbers_in_args(args)
    _add_default_args(parser, ["deorbit", "energies", "loglevel", "debug"])

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)

    event_list = load_events(args.fname)
    emin = emax = None
    if args.energy_interval is not None:
        emin, emax = args.energy_interval
        event_list, elabel = filter_energy(event_list, emin, emax)
        if elabel != "Energy":
            raise ValueError(
                "You are filtering by energy but the data are not calibrated"
            )

    new_event_list = scramble(
        event_list,
        smooth_kind=args.smooth_kind,
        dt=args.dt,
        pulsed_fraction=args.pulsed_fraction,
        deadtime=args.deadtime,
        orbit_par=args.deorbit_par,
        frequency=args.frequency,
    )

    if args.outfile is not None:
        outfile = args.outfile
    else:
        label = "_scramble"

        if args.smooth_kind == "pulsed":
            label += f"_pulsed_df{args.pulsed_fraction:g}"
        elif args.smooth_kind == "smooth":
            label += f"_smooth_dt{args.dt:g}s"
        if args.deadtime > 0:
            label += f"_deadtime_{args.deadtime:g}"
        if args.energy_interval is not None:
            label += f"_{emin:g}-{emax:g}keV"

        outfile = args.fname.replace(
            HEN_FILE_EXTENSION, f"{label}" + HEN_FILE_EXTENSION
        )
    save_events(new_event_list, outfile)
    return outfile


def main(args=None):
    """Main function called by the `HENfake` command line script."""
    import argparse
    from .base import _add_default_args, check_negative_numbers_in_args

    description = (
        "Create an event file in FITS format from an event list, or simulating"
        " it. If input event list is not specified, generates the events "
        "randomly"
    )
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "-e",
        "--event-list",
        type=str,
        default=None,
        help="File containing event list",
    )
    parser.add_argument(
        "-l",
        "--lc",
        type=str,
        default=None,
        help="File containing light curve",
    )
    parser.add_argument(
        "-c",
        "--ctrate",
        type=float,
        default=None,
        help="Count rate for simulated events",
    )
    parser.add_argument(
        "-o",
        "--outname",
        type=str,
        default="events.evt",
        help="Output file name",
    )
    parser.add_argument(
        "-i", "--instrument", type=str, default=None, help="Instrument name"
    )
    parser.add_argument("-m", "--mission", type=str, default=None, help="Mission name")
    parser.add_argument(
        "--tstart",
        type=float,
        default=None,
        help="Start time of the observation (s from MJDREF)",
    )
    parser.add_argument(
        "--tstop",
        type=float,
        default=None,
        help="End time of the observation (s from MJDREF)",
    )
    parser.add_argument(
        "--mjdref",
        type=float,
        default=55197.00076601852,
        help="Reference MJD",
    )
    parser.add_argument(
        "--deadtime",
        type=float,
        default=None,
        nargs="+",
        help="Dead time magnitude. Can be specified as a "
        "single number, or two. In this last case, the "
        "second value is used as sigma of the dead time "
        "distribution",
    )
    args = check_negative_numbers_in_args(args)
    _add_default_args(parser, ["loglevel", "debug"])

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)
    with log.log_to_file("HENfake.log"):
        additional_columns = {}
        livetime = None
        if args.lc is None and args.ctrate is None and args.event_list is not None:
            event_list = _read_event_list(args.event_list)
        elif args.lc is not None or args.ctrate is not None:
            event_list = EventList()
            if args.lc is not None:
                lc = _read_light_curve(args.lc)
            elif args.ctrate is not None:
                tstart = assign_value_if_none(args.tstart, 0)
                tstop = assign_value_if_none(args.tstop, 1024)
                dt = (tstop - tstart) / 1024
                t = np.arange(tstart, tstop + 1, dt)
                lc = Lightcurve(time=t, counts=args.ctrate + np.zeros_like(t), dt=dt)
            event_list.simulate_times(lc)
            nevents = len(event_list.time)
            event_list.pi = np.zeros(nevents, dtype=int)
            event_list.mjdref = args.mjdref
            log.info("{} events generated".format(nevents))
        else:
            event_list = None

        if args.deadtime is not None and event_list is not None:
            deadtime = args.deadtime[0]
            deadtime_sigma = None
            if len(args.deadtime) > 1:
                deadtime_sigma = args.deadtime[1]
            event_list, info = filter_for_deadtime(
                event_list, deadtime, dt_sigma=deadtime_sigma, return_all=True
            )

            log.info("{} events after filter".format(len(event_list.time)))

            prior = np.zeros_like(event_list.time)

            prior[1:] = np.diff(event_list.time) - info.deadtime[:-1]

            additional_columns["PRIOR"] = {"data": prior, "format": "D"}
            additional_columns["KIND"] = {
                "data": info.is_event,
                "format": "L",
            }
            livetime = np.sum(prior)

        generate_fake_fits_observation(
            event_list=event_list,
            filename=args.outname,
            instr=args.instrument,
            mission=args.mission,
            tstart=args.tstart,
            tstop=args.tstop,
            mjdref=args.mjdref,
            livetime=livetime,
            additional_columns=additional_columns,
        )
