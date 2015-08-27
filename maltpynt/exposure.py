# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate the exposure correction for light curves. Only works for data
   taken in specific data modes of NuSTAR, where all events are telemetered.

"""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
from .read_events import load_events_and_gtis
from .io import get_file_type, save_lcurve, MP_FILE_EXTENSION
from .base import create_gti_mask, mp_root
import logging


def get_livetime_per_bin(times, events, priors, dt=None, gti=None):
    """Get the livetime in a series of time intervals.

    Parameters
    ----------
    times : array-like
        The array of times to look at
    events : array-like
        A list of events, producing dead time
    priors : array-like
        The livetime before each event (as in the PRIOR column of unfiltered
        NuSTAR event files)

    Returns
    -------
    livetime_array : array-like
        An array of the same length as times, containing the live time values

    Other Parameters
    ----------------
    dt : float or array-like
        The width of the time bins of the time array. Can be a single float or
        an array of the same length as times
    gti : [[g0_0, g0_1], [g1_0, g1_1], ...]
         Good time intervals. Defaults to
         [[time[0] - dt[0]/2, time[-1] + dt[-1]/2]]

    """

    assert len(events) == len(priors), \
        "`events` and `priors` must be of the same length"

    if dt is None:
        dt = np.median(np.diff(times))
    try:
        len(dt)
    except:
        dt = dt + np.zeros(len(times))

    ev_fl = np.array(events - events[0], dtype=np.float64)
    pr_fl = np.array(priors, dtype=np.float64)
    livetime_starts = ev_fl - pr_fl

    tbins = np.array(
        np.append(times - dt / 2, [times[-1] + dt[-1] / 2]) - events[0],
        dtype=np.float64)

    tbin_starts = tbins[:-1]

    livetime_array = np.zeros_like(times)

    # Calculate live time.
    lts_bin = np.searchsorted(tbin_starts, livetime_starts, 'right') - 1
    ev_bin = np.searchsorted(tbin_starts, ev_fl, 'right') - 1
    # First of all, just consider livetimes inside bin borders.

    first_pass = ev_bin == lts_bin

    expo, bins = np.histogram(ev_fl[first_pass], bins=tbins,
                              weights=pr_fl[first_pass])

    assert np.all(expo) >= 0, expo
    livetime_array += expo
    max_bin_diff = np.max(ev_bin - lts_bin)

    # Now, overlapping
    for bin_diff in range(max_bin_diff, 0, -1):
        idxs = ev_bin == lts_bin + bin_diff
        # Filter only events relevant to this case
        ev_bin_good = ev_bin[idxs]
        lts_bin_good = lts_bin[idxs]
        ev_good = ev_fl[idxs]
        lt_good = livetime_starts[idxs]

        # find corresponding time bins
        e_idx = np.searchsorted(tbin_starts, ev_good, 'right') - 1
        _tbins = tbin_starts[e_idx]
        livetime_array[ev_bin_good] += ev_good - _tbins
        assert np.all(ev_good - _tbins >= 0), \
            "Invalid boundaries. Contact the developer: {}".format(
                ev_good - _tbins)

        l_idx = np.searchsorted(tbin_starts, lt_good, 'right')
        _tbins = tbin_starts[l_idx]
        livetime_array[lts_bin_good] += _tbins - lt_good
        assert np.all(_tbins - lt_good >= 0), \
            "Invalid boundaries. Contact the developer: {}".format(
                _tbins - lt_good)

        print(ev_good - lt_good, pr_fl[idxs])
        # TODO: add bins in the middle if max_bin_diff > 1
        # Complete bins
        if bin_diff > 1:
            for i in range(bin_diff):
                livetime_array[lts_bin_good + bin_diff] += \
                    dt[lts_bin_good + bin_diff]
    return livetime_array


def _plot_dead_time_from_uf(uf_file, outroot="expo"):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from numpy import histogram

    additional_columns = ["PRIOR", "GRADE", "PI", "X", "Y", "SHIELD",
                          "SHLD_T", "SHLD_HI"]
    events, gtis, additional, tstart, tstop = \
        load_events_and_gtis(uf_file,
                             additional_columns=additional_columns,
                             return_limits=True)

    priors = additional["PRIOR"]

    dead_times = np.diff(events) - priors[1:]
    shields = additional["SHIELD"][1:]
    shld_t = additional["SHLD_T"][1:]
    shld_hi = additional["SHLD_HI"][1:]

    bins = np.percentile(dead_times, np.linspace(0, 100, 1000))
    hist_all, bins_all = histogram(dead_times, bins=bins, density=True)
    hist_shield, bins_shield = histogram(dead_times[shields > 0], bins=bins,
                                         density=True)
    hist_noshield, bins_noshield = histogram(dead_times[shields == 0],
                                             bins=bins, density=True)
    hist_shld_hi, bins_shld_hi = histogram(dead_times[shld_hi > 0],
                                           bins=bins, density=True)

    bin_centers = bins[:-1] + np.diff(bins) / 2
    fig = plt.figure("Dead time distribution", figsize=(10, 10))
    gs = GridSpec(2, 1, hspace=0)
    ax1 = plt.subplot(gs[0])
    ax1.loglog(bin_centers, hist_all, drawstyle="steps-mid", label="all")
    ax1.loglog(bin_centers, hist_shield, drawstyle="steps-mid", label="shield")
    ax1.loglog(bin_centers, hist_shld_hi, drawstyle="steps-mid",
               label="shld_hi")
    ax1.loglog(bin_centers, hist_noshield, drawstyle="steps-mid",
               label="no shield")
    ax1.set_ylabel("Occurrences (arbitrary units)")
    ax1.legend()
    ax2 = plt.subplot(gs[1], sharex=ax1)

    for sht in set(shld_t[shld_t > 0]):
        hs, bs = histogram(dead_times[shld_t == sht], bins=bins, density=True)
        ax2.loglog(bin_centers, hs, drawstyle="steps-mid",
                   label="shield time {}".format(sht))
    ax2.set_xlabel("Dead time (s)")
    ax2.set_ylabel("Occurrences (arbitrary units)")
    ax2.legend()
    plt.draw()
    fig.savefig(outroot + "_deadt_distr.png")


def get_exposure_from_uf(time, uf_file, dt=None, gti=None):
    """Get livetime from unfiltered event file.

    Parameters
    ----------
    time : array-like
        The time bins of the light curve
    uf_file : str
        Unfiltered event file (the one in the event_cl directory with the _uf
        suffix)

    Returns
    -------
    expo : array-like
        Exposure (livetime) values corresponding to time bins

    Other Parameters
    ----------------
    dt : float
        If time array is not sampled uniformly, dt can be specified here.

    """

    if dt is None:
        dt = np.min(np.diff(time))

    additional_columns = ["PRIOR", "GRADE", "PI", "X", "Y"]
    events, gtis, additional, tstart, tstop = \
        load_events_and_gtis(uf_file,
                             additional_columns=additional_columns,
                             return_limits=True)

    priors = additional["PRIOR"]
    # grade = additional["GRADE"]
    # pis = additional["PI"]
    # xs = additional["X"]
    # ys = additional["Y"]
    #
    # filt = (grade < 32) & (pis >= 0) & (x is not None) & (y is not None)

    expo = get_livetime_per_bin(time, events, priors, dt, gti=gti)
    return expo


def _plot_corrected_light_curve(time, lc, expo, gti=None, outroot="expo"):
    import matplotlib.pyplot as plt
    good = create_gti_mask(time, gti)
    fig = plt.figure("Exposure-corrected lc")
    plt.plot(time[good], expo[good] / np.max(expo) * np.max(lc[good]),
             label="Exposure (arbitrary units)", zorder=10)
    plt.plot(time[good], lc[good], label="Light curve", zorder=20)
    plt.plot(time[good], lc[good] / expo[good],
             label="Exposure-corrected Light curve")
    plt.legend()
    fig.savefig(outroot + "_corr_lc.png")


def correct_lightcurve(lc_file, uf_file, outname=None):
    """Apply exposure correction to light curve.

    Parameters
    ----------
    lc_file : str
        The light curve file, in MaLTPyNT format
    uf_file : str
        The unfiltered event file, in FITS format

    Returns
    -------
    outdata : str
        Output data structure

    Other Parameters
    ----------------
    outname : str
        Output file name
    """
    if outname is None:
        outroot = mp_root(lc_file)
        outname = outroot + "_lccorr" + MP_FILE_EXTENSION

    ftype, contents = get_file_type(lc_file)

    time = contents["time"]
    lc = contents["lc"]
    dt = contents["dt"]
    gti = contents["GTI"]

    expo = get_exposure_from_uf(time, uf_file, dt=dt, gti=gti)

    outdata = contents.copy()

    outdata["lc"] = lc / expo
    outdata["expo"] = expo

    save_lcurve(outdata, outname)
    return outdata


def main(args=None):
    import argparse
    description = (
        'Create exposure light curve based on unfiltered event files.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("lcfile", help="Light curve file (MaltPyNT format)")
    parser.add_argument("uffile", help="Unfiltered event file (FITS)")
    parser.add_argument("-o", "--outroot", type=str, default=None,
                        help='Root of output file names')

    parser.add_argument("--loglevel",
                        help=("use given logging level (one between INFO, "
                              "WARNING, ERROR, CRITICAL, DEBUG; "
                              "default:WARNING)"),
                        default='WARNING',
                        type=str)
    parser.add_argument("--debug", help="use DEBUG logging level",
                        default=False, action='store_true')
    parser.add_argument("--plot", help="Plot on window",
                        default=False, action='store_true')

    args = parser.parse_args(args)

    if args.debug:
        args.loglevel = 'DEBUG'

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(filename='MPexposure.log', level=numeric_level,
                        filemode='w')

    lc_file = args.lcfile
    uf_file = args.uffile

    outroot = args.outroot
    if outroot is None:
        outroot = mp_root(lc_file)

    outname = outroot + "_lccorr" + MP_FILE_EXTENSION

    outdata = correct_lightcurve(lc_file, uf_file, outname)

    time = outdata["time"]
    lc = outdata["lc"]
    expo = outdata["expo"]
    gti = outdata["GTI"]

    _plot_corrected_light_curve(time, lc * expo, expo, gti, outroot)

    _plot_dead_time_from_uf(uf_file, outroot)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.show()
