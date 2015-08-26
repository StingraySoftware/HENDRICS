# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate the exposure correction for light curves. Only works for data
   taken in specific data modes of NuSTAR, where all events are telemetered.

"""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
from .read_events import load_events_and_gtis
from .io import get_file_type
from .base import create_gti_mask


def plot_dead_time_from_uf(uf_file):
    import matplotlib.pyplot as plt
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

    plt.figure("Dead time distribution")
    bins = np.percentile(dead_times, np.linspace(0, 100, 1000))
    hist_all, bins_all = histogram(dead_times, bins=bins, density=True)
    hist_shield, bins_shield = histogram(dead_times[shields > 0], bins=bins,
                                         density=True)
    hist_noshield, bins_noshield = histogram(dead_times[shields == 0],
                                             bins=bins, density=True)
    hist_shld_hi, bins_shld_hi = histogram(dead_times[shld_hi > 0],
                                           bins=bins, density=True)

    bin_centers = bins[:-1] + np.diff(bins) / 2
    plt.loglog(bin_centers, hist_all, drawstyle="steps-mid", label="all")
    plt.loglog(bin_centers, hist_shield, drawstyle="steps-mid", label="shield")
    plt.loglog(bin_centers, hist_shld_hi, drawstyle="steps-mid",
               label="shld_hi")
    plt.loglog(bin_centers, hist_noshield, drawstyle="steps-mid",
               label="no shield")
    for sht in set(shld_t[shld_t > 0]):
        hs, bs = histogram(dead_times[shld_t == sht], bins=bins, density=True)
        plt.loglog(bin_centers, hs, drawstyle="steps-mid",
                   label="shield time {}".format(sht))
    plt.legend()
    plt.show()


def get_exposure_from_uf(time, uf_file, dt=None):
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

    tbins = np.append(time - dt / 2, [time[-1] + dt / 2])

    # This is wrong. Doesn't consider time bin borders.
    expo, bins = np.histogram(events, bins=tbins, weights=priors)

    return expo


def main(args=None):
    import argparse
    import matplotlib.pyplot as plt
    description = (
        'Create exposure light curve based on unfiltered event files.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("lcfile", help="Light curve file (MaltPyNT format)")
    parser.add_argument("uffile", help="Unfiltered event file (FITS)")

    args = parser.parse_args(args)

    lc_file = args.lcfile
    uf_file = args.uffile
    ftype, contents = get_file_type(lc_file)

    time = contents["time"]
    lc = contents["lc"]
    dt = contents["dt"]
    gti = contents["GTI"]
    expo = get_exposure_from_uf(time, uf_file, dt=dt)

    good = create_gti_mask(time, gti)

    plt.plot(time[good], expo[good] / np.max(expo) * np.max(lc[good]),
             label="Exposure (arbitrary units)")
    plt.plot(time[good], lc[good], label="Light curve")
    plt.plot(time[good], lc[good] / expo[good],
             label="Exposure-corrected Light curve")
    plt.legend()
    plt.show()
    plot_dead_time_from_uf(uf_file)
