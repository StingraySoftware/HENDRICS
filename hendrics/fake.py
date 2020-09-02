# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to simulate data and produce a fake event file."""

import os
import warnings
import copy
import numpy as np
import numpy.random as ra
from astropy import log
from astropy.logger import AstropyUserWarning
from stingray.events import EventList
from stingray.lightcurve import Lightcurve
from stingray.utils import assign_value_if_none
from .io import get_file_format, load_lcurve
from .io import load_events, save_events, HEN_FILE_EXTENSION
from .base import _empty, r_in

from .lcurve import lcurve_from_fits
from .base import njit


def _paralyzable_dead_time(event_list, dead_time):
    mask = np.ones(len(event_list), dtype=bool)
    dead_time_end = event_list + dead_time
    bad = dead_time_end[:-1] > event_list[1:]
    # Easy: paralyzable case. Here, events coming during dead time produce
    # more dead time. So...
    mask[1:][bad] = False

    return event_list[mask], mask


@njit()
def _nonpar_core(event_list, dead_time_end, mask):
    for i in range(1, len(event_list)):
        if event_list[i] < dead_time_end[i - 1]:
            dead_time_end[i] = dead_time_end[i - 1]
            mask[i] = False
    return mask


def _non_paralyzable_dead_time(event_list, dead_time):
    event_list_dbl = (event_list - event_list[0]).astype(np.double)
    dead_time_end = event_list_dbl + np.double(dead_time)
    mask = np.ones(event_list_dbl.size, dtype=bool)
    mask = _nonpar_core(event_list_dbl, dead_time_end, mask)
    return event_list[mask], mask


def filter_for_deadtime(
    event_list,
    deadtime,
    bkg_ev_list=None,
    dt_sigma=None,
    paralyzable=False,
    additional_data=None,
    return_all=False,
):
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
    if not isinstance(event_list, EventList):
        event_list_obj = EventList(event_list)
    else:
        event_list_obj = event_list

    ev_list = event_list_obj.time

    if deadtime <= 0.0:
        return copy.copy(event_list)

    # Create the total lightcurve, and a "kind" array that keeps track
    # of the events classified as "signal" (True) and "background" (False)
    if bkg_ev_list is not None:
        tot_ev_list = np.append(ev_list, bkg_ev_list)
        ev_kind = np.append(
            np.ones(len(ev_list), dtype=bool),
            np.zeros(len(bkg_ev_list), dtype=bool),
        )
        order = np.argsort(tot_ev_list)
        tot_ev_list = tot_ev_list[order]
        ev_kind = ev_kind[order]
        del order
    else:
        tot_ev_list = ev_list
        ev_kind = np.ones(len(ev_list), dtype=bool)

    nevents = len(tot_ev_list)
    all_ev_kind = ev_kind.copy()

    if dt_sigma is not None:
        deadtime_values = ra.normal(deadtime, dt_sigma, nevents)
    else:
        deadtime_values = np.zeros(nevents) + deadtime

    initial_len = len(tot_ev_list)

    if paralyzable:
        tot_ev_list, saved_mask = _paralyzable_dead_time(
            tot_ev_list, deadtime_values
        )

    else:
        tot_ev_list, saved_mask = _non_paralyzable_dead_time(
            tot_ev_list, deadtime_values
        )

    ev_kind = ev_kind[saved_mask]
    deadtime_values = deadtime_values[saved_mask]
    final_len = len(tot_ev_list)
    log.info(
        "filter_for_deadtime: "
        "{0}/{1} events rejected".format(initial_len - final_len, initial_len)
    )
    retval = EventList(time=tot_ev_list[ev_kind], mjdref=event_list_obj.mjdref)

    if hasattr(event_list_obj, "pi") and event_list_obj.pi is not None:
        warnings.warn(
            "PI information is lost during dead time filtering",
            AstropyUserWarning,
        )

    if not isinstance(event_list, EventList):
        retval = retval.time

    if return_all:
        additional_output.uf_events = tot_ev_list
        additional_output.is_event = ev_kind
        additional_output.deadtime = deadtime_values
        additional_output.mask = saved_mask[all_ev_kind]
        additional_output.bkg = tot_ev_list[np.logical_not(ev_kind)]
        retval = [retval, additional_output]

    return retval


def generate_fake_fits_observation(
    event_list=None,
    filename=None,
    instr="FPMA",
    gti=None,
    tstart=None,
    tstop=None,
    mission="NUSTAR",
    mjdref=55197.00076601852,
    livetime=None,
    additional_columns={},
):
    """Generate fake NuSTAR data.

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

    if event_list is None:
        tstart = assign_value_if_none(tstart, 8e7)
        tstop = assign_value_if_none(tstop, tstart + 1025)
        ev_list = sorted(ra.uniform(tstart, tstop, 1000))
    else:
        ev_list = event_list.time

    if hasattr(event_list, "pi") and event_list.pi is not None:
        pi = event_list.pi
    else:
        pi = ra.randint(0, 1024, len(ev_list))

    if hasattr(event_list, "cal_pi") and event_list.cal_pi is not None:
        cal_pi = event_list.cal_pi
    else:
        cal_pi = pi / 3

    tstart = assign_value_if_none(tstart, np.floor(ev_list[0]))
    tstop = assign_value_if_none(tstop, np.ceil(ev_list[-1]))
    gti = assign_value_if_none(gti, np.array([[tstart, tstop]]))
    filename = assign_value_if_none(filename, "events.evt")
    livetime = assign_value_if_none(livetime, tstop - tstart)

    if livetime > tstop - tstart:
        raise ValueError(
            "Livetime must be equal or smaller than " "tstop - tstart"
        )

    # Create primary header
    prihdr = fits.Header()
    prihdr["OBSERVER"] = "Edwige Bubble"
    prihdr["TELESCOP"] = (mission, "Telescope (mission) name")
    prihdr["INSTRUME"] = (instr, "Instrument name")
    prihdu = fits.PrimaryHDU(header=prihdr)

    # Write events to table
    col1 = fits.Column(name="TIME", format="1D", array=ev_list)

    allcols = [col1]

    if mission.lower().strip() == "xmm":
        ccdnr = np.zeros(len(ev_list)) + 1
        ccdnr[1] = 2  # Make it less trivial
        ccdnr[10] = 7
        allcols.append(fits.Column(name="CCDNR", format="1J", array=ccdnr))

    if mission.lower().strip() in ["xmm", "swift"]:
        allcols.append(fits.Column(name="PHA", format="1J", array=pi))
        allcols.append(fits.Column(name="PI", format="1J", array=cal_pi))
    else:
        allcols.append(fits.Column(name="PI", format="1J", array=pi))

    for c in additional_columns.keys():
        col = fits.Column(
            name=c,
            array=additional_columns[c]["data"],
            format=additional_columns[c]["format"],
        )
        allcols.append(col)

    cols = fits.ColDefs(allcols)
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name = "EVENTS"

    # ---- Fake lots of information ----
    tbheader = tbhdu.header
    tbheader["OBSERVER"] = "Edwige Bubble"
    tbheader["COMMENT"] = (
        "FITS (Flexible Image Transport System) format is"
        " defined in 'Astronomy and Astrophysics', volume"
        " 376, page 359; bibcode: 2001A&A...376..359H"
    )
    tbheader["TELESCOP"] = (mission, "Telescope (mission) name")
    tbheader["INSTRUME"] = (instr, "Instrument name")
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
    tbheader["MJDREFI"] = (
        int(mjdref),
        "TDB time reference; Modified Julian Day (int)",
    )
    tbheader["MJDREFF"] = (
        mjdref - int(mjdref),
        "TDB time reference; Modified Julian Day (frac)",
    )
    tbheader["TIMEREF"] = (
        "SOLARSYSTEM",
        "Times are pathlength-corrected to barycenter",
    )
    tbheader["CLOCKAPP"] = (False, "TRUE if timestamps corrected by gnd sware")
    tbheader["COMMENT"] = (
        "MJDREFI+MJDREFF = epoch of Jan 1, 2010, in TT " "time system."
    )
    tbheader["TIMEUNIT"] = ("s", "unit for time keywords")
    tbheader["TSTART"] = (
        tstart,
        "Elapsed seconds since MJDREF at start of file",
    )
    tbheader["TSTOP"] = (tstop, "Elapsed seconds since MJDREF at end of file")
    tbheader["LIVETIME"] = (livetime, "On-source time")
    tbheader["TIMEZERO"] = (0.000000e00, "Time Zero")
    tbheader["COMMENT"] = "Generated with HENDRICS by {0}".format(
        os.getenv("USER")
    )

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
        gtinames = ["STDGTI01", "STDGTI02", "STDGTI07"]

    all_new_hdus = [prihdu, tbhdu]
    for name in gtinames:
        gtihdu = fits.BinTableHDU.from_columns(cols)
        gtihdu.name = name
        all_new_hdus.append(gtihdu)

    thdulist = fits.HDUList(all_new_hdus)

    thdulist.writeto(filename, overwrite=True)
    return thdulist


def _read_event_list(filename):
    if filename is not None:
        warnings.warn(
            "Input event lists not yet implemented", AstropyUserWarning
        )
    return None, None


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
    event_list, smooth_kind="flat", dt=None, pulsed_fraction=0.0, deadtime=0.0
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

    for (i_start, i_stop), gti_boundary in zip(idxs, new_event_list.gti):
        locally_flat = False
        nevents = i_stop - i_start
        t_start, t_stop = gti_boundary[0], gti_boundary[1]
        if nevents < 1:
            continue
        length = t_stop - t_start
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
            new_event_list.time[i_start:i_stop] = new_events[
                : i_stop - i_start
            ]
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
            df = 1 / length
            # Frequency is one, but can be anywhere in the frequency bin (for
            # sensitivity losses)
            frequency = 1 + np.random.uniform(-df, df)
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
            sinusoid = pulsed_fraction / 2 * np.sin(np.pi * 2 * times)

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
        "--outfile", type=str, default=None, help="Output file name"
    )
    args = check_negative_numbers_in_args(args)
    _add_default_args(parser, ["loglevel", "debug"])

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)

    event_list = load_events(args.fname)
    new_event_list = scramble(
        event_list,
        smooth_kind=args.smooth_kind,
        dt=args.dt,
        pulsed_fraction=args.pulsed_fraction,
        deadtime=args.deadtime,
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

        outfile = args.fname.replace(
            HEN_FILE_EXTENSION, "_scramble" + HEN_FILE_EXTENSION
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
        "-i", "--instrument", type=str, default="FPMA", help="Instrument name"
    )
    parser.add_argument(
        "-m", "--mission", type=str, default="NUSTAR", help="Mission name"
    )
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
        "--mjdref", type=float, default=55197.00076601852, help="Reference MJD"
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
        if (
            args.lc is None
            and args.ctrate is None
            and args.event_list is not None
        ):
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
                lc = Lightcurve(
                    time=t, counts=args.ctrate + np.zeros_like(t), dt=dt
                )
            event_list.simulate_times(lc)
            nevents = len(event_list.time)
            event_list.pi = np.zeros(nevents, dtype=int)
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
            additional_columns["KIND"] = {"data": info.is_event, "format": "L"}
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
