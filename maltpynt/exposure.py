# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate the exposure correction for light curves. Only works for data
   taken in specific data modes of NuSTAR, where all events are telemetered.

"""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
from .read_events import load_events_and_gtis
from .io import get_file_type


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

    lc_file = args.lc
    uf_file = args.uf
    ftype, contents = get_file_type(lc_file)

    time = contents["time"]
    lc = contents["lc"]
    dt = contents["dt"]
    expo = get_exposure_from_uf(time, uf_file, dt=dt)

    plt.plot(time, expo / np.max(expo) * np.max(lc))
    plt.plot(time, lc)
    plt.plot(time, lc / expo)
    plt.show()
