# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate the exposure correction for light curves. Only works for data
   taken in specific data modes of NuSTAR, where all events are telemetered.

"""
import numpy as np
from .mp_read_events import mp_load_events_and_gtis
from .mp_io import mp_load_data


def mp_get_exposure_from_uf(time, uf_file, dt=None):
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
        mp_load_events_and_gtis(uf_file,
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

    expo = np.histogram(priors, bins=tbins)

    return expo


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    lc_file = sys.argv[1]
    uf_file = sys.argv[2]

    contents, ftype = mp_load_data(lc_file)
    time = contents["time"]
    dt = contents["dt"]
    expo = mp_get_exposure_from_uf(time, uf_file, dt=dt)

    plt.plot(time, expo)
    plt.show()
