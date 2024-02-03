"""Save different input files in PRESTO-readable format."""

from astropy import log
from astropy.coordinates import SkyCoord
import numpy as np
from .io import high_precision_keyword_read, get_file_type, HEN_FILE_EXTENSION
from .base import deorbit_events, interpret_bintime

MAXBIN = 100000000


def get_header_info(obj):
    """Get header info from a Stingray object."""
    from astropy.io.fits import Header

    header = Header.fromstring(obj.header)
    info = type("", (), {})()
    info.mjdref = high_precision_keyword_read(header, "MJDREF")
    info.telescope = header["TELESCOP"]
    info.instrument = header["INSTRUME"]
    info.source = header["OBJECT"]
    try:
        user = header["USER"]
    except KeyError:
        user = "Unknown"
    info.observer = user
    info.user = user
    info.tstart = header["TSTART"]
    info.tstop = header["TSTOP"]
    try:
        ra = header["RA_OBJ"]
        dec = header["DEC_OBJ"]
    except KeyError:
        ra = header["RA_PNT"]
        dec = header["DEC_PNT"]

    a = SkyCoord(ra, dec, unit="degree")
    info.raj = (
        (a.ra.to_string("hourangle"))
        .replace("s", "")
        .replace("h", ":")
        .replace("m", ":")
    )
    info.decj = (a.dec.to_string()).replace("s", "").replace("d", ":").replace("m", ":")
    if hasattr(obj, "e_interval"):
        e0, e1 = obj.e_interval
    elif hasattr(obj, "energy") and obj.energy is not None:
        e0, e1 = np.min(obj.energy), np.max(obj.energy)
    else:
        e0, e1 = 0, 0
    info.centralE = (e0 + e1) / 2
    info.bandpass = e1 - e0

    return info


def _save_to_binary(lc, filename):
    """Save a light curve to binary format."""
    nc = len(lc.counts)
    lc.counts[: nc // 2 * 2].astype("float32").tofile(filename)
    return


def save_lc_to_binary(lc, filename):
    """Save a light curve to binary format.

    Parameters
    ----------
    lc : `:class:stingray.Lightcurve`
        Input light curve
    filename : str
        Output file name
    Returns
    -------
    lcinfo : object
        light curve info
    """
    tstart = lc.tstart
    tstop = lc.tstart + lc.tseg
    nbin = lc.n
    bin_time = lc.dt
    _save_to_binary(lc, filename)

    lcinfo = type("", (), {})()
    lcinfo.bin_intervals_start = [0]
    lcinfo.bin_intervals_stop = [nbin]
    lcinfo.lclen = nbin
    lcinfo.tstart = tstart
    lcinfo.dt = bin_time
    lcinfo.tseg = tstop - tstart
    lcinfo.nphot = np.sum(lc.counts)
    return lcinfo


def save_events_to_binary(
    events, filename, bin_time, tstart=None, emin=None, emax=None
):
    """Save an event list to binary format.

    Parameters
    ----------
    events : `:class:stingray.Eventlist`
        Input event list
    filename : str
        Output file name
    bin_time : float
        Bin time of the output light curve

    Other parameters
    ----------------
    tstart : float
        Starting time
    emin : float
        Minimum energy of the photons
    emax : float
        Maximum energy of the photons

    Returns
    -------
    lcinfo : object
        light curve info
    """
    import struct

    if tstart is None:
        tstart = events.gti[0, 0]

    if emin is not None and emax is not None:
        if not hasattr(events, "energy") or events.energy is None:
            raise ValueError(
                "Energy filtering requested for uncalibrated event " "list"
            )

        good = (events.energy >= emin) & (events.energy < emax)
        events = events.apply_mask(good)
        # events.time = events.time[good]

    tstop = events.gti[-1, 1]
    nbin = (tstop - tstart) / bin_time

    lclen = 0
    file = open(filename, "wb")
    nphot = 0
    for i in np.arange(0, nbin, MAXBIN):
        t0 = i * bin_time + tstart

        lastbin = int(np.min([MAXBIN, (nbin - i) // 4 * 4]))
        t1 = t0 + lastbin * bin_time

        good = (events.time >= t0) & (events.time < t1)

        goodev = events.time[good]
        hist, times = np.histogram(goodev, bins=np.linspace(t0, t1, lastbin + 1))

        lclen += lastbin
        s = struct.pack("f" * len(hist), *hist)
        file.write(s)
        nphot += len(goodev)
    file.close()

    lcinfo = type("", (), {})()
    lcinfo.bin_intervals_start = np.floor((events.gti[:, 0] - tstart) / bin_time)
    lcinfo.bin_intervals_stop = np.floor((events.gti[:, 1] - tstart) / bin_time)
    lcinfo.lclen = lclen
    lcinfo.tstart = tstart
    lcinfo.dt = bin_time
    lcinfo.tseg = tstop - tstart
    lcinfo.nphot = nphot
    return lcinfo


def save_inf(lcinfo, info, filename):
    """Save information file."""

    lclen = lcinfo.lclen
    bin_intervals_start, bin_intervals_stop = (
        lcinfo.bin_intervals_start,
        lcinfo.bin_intervals_stop,
    )

    epoch = info.mjdref + lcinfo.tstart / 86400

    with open(filename, "w") as f:
        print(
            " Data file name without suffix         "
            " =  {}".format(filename.replace(".inf", "")),
            file=f,
        )
        print(
            " Telescope used                        " " =  {}".format(info.telescope),
            file=f,
        )
        print(
            " Instrument used                       " " =  {}".format(info.instrument),
            file=f,
        )
        print(
            " Object being observed                 " " =  {}".format(info.source),
            file=f,
        )
        print(
            " J2000 Right Ascension (hh:mm:ss.ssss) " " =  {}".format(info.raj),
            file=f,
        )
        print(
            " J2000 Declination     (dd:mm:ss.ssss) " " =  {}".format(info.decj),
            file=f,
        )
        print(
            " Data observed by                      " " =  {}".format(info.observer),
            file=f,
        )
        print(
            " Epoch of observation (MJD)            " " =  {:05.15f}".format(epoch),
            file=f,
        )
        print(" Barycentered?           (1=yes, 0=no) " " =  1", file=f)
        print(
            " Number of bins in the time series     " " =  {lclen}".format(lclen=lclen),
            file=f,
        )
        print(
            " Width of each time series bin (sec)   "
            " =  {bintime}".format(bintime=lcinfo.dt),
            file=f,
        )
        print(" Any breaks in the data? (1 yes, 0 no) " " =  1", file=f)
        for i, st in enumerate(bin_intervals_start):
            print(
                " On/Off bin pair # {ngti:>2}                  "
                " =  {binstart:<11}, "
                "{binstop:<11}".format(
                    ngti=i + 1, binstart=st, binstop=bin_intervals_stop[i]
                ),
                file=f,
            )
        print(" Type of observation (EM band)         " " =  X-ray", file=f)
        print(" Field-of-view diameter (arcsec)       " " =  400", file=f)
        print(
            " Central energy (kev)                  " " =  {}".format(info.centralE),
            file=f,
        )
        print(
            " Energy bandpass (kev)                 " " =  {}".format(info.bandpass),
            file=f,
        )
        print(
            " Data analyzed by                      " " =  {}".format(info.user),
            file=f,
        )
        print(" Any additional notes:", file=f)
        print(
            "       T = {length}, Nphot={nphot}".format(
                length=lcinfo.tseg, nphot=lcinfo.nphot
            ),
            file=f,
        )

    return


def main_presto(args=None):
    import argparse
    from .base import _add_default_args, check_negative_numbers_in_args

    description = "Save light curves in a format readable to PRESTO"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("files", help="List of input light curves", nargs="+")

    parser.add_argument(
        "-l",
        "--max-length",
        help="Maximum length of light " "curves (split otherwise)",
        type=np.longdouble,
        default=1e32,
    )

    args = check_negative_numbers_in_args(args)
    _add_default_args(
        parser,
        ["bintime", "energies", "deorbit", "nproc", "loglevel", "debug"],
    )

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = "DEBUG"

    log.setLevel(args.loglevel)

    bintime = np.longdouble(interpret_bintime(args.bintime))

    if args.energy_interval is None:
        args.energy_interval = [None, None]
    with log.log_to_file("HENbinary.log"):
        for f in args.files:
            print(f)
            outfile = f.replace(HEN_FILE_EXTENSION, ".dat")
            ftype, contents = get_file_type(f)
            if ftype == "lc":
                lcinfo = save_lc_to_binary(contents, outfile)
            elif ftype == "events":
                if args.deorbit_par is not None:
                    contents = deorbit_events(contents, args.deorbit_par)
                lcinfo = save_events_to_binary(
                    contents,
                    outfile,
                    bin_time=bintime,
                    emin=args.energy_interval[0],
                    emax=args.energy_interval[1],
                )
            else:
                raise ValueError("File type not recognized")

            info = get_header_info(contents)
            save_inf(lcinfo, info, f.replace(HEN_FILE_EXTENSION, ".inf"))
