from __future__ import division, print_function
import numpy as np
from mp_base import mp_create_gti_mask, mp_create_gti_from_condition


def mp_cross_gtis(gti_list, bin_time=1, debug=True):
    ninst = len(gti_list)
    if ninst == 1:
        return gti_list[0]

    start = np.min([g[0][0] for g in gti_list])
    stop = np.max([g[-1][-1] for g in gti_list])

    times = np.arange(start, stop, 1, dtype=np.longdouble)

    mask0 = mp_create_gti_mask(times, gti_list[0], verbose=0)

    if debug:
        step = 1
        colors = ['k', 'r', 'g', 'b']
        colors += colors + colors + colors
        import matplotlib.pyplot as plt
        for g in gti_list[0]:
            plt.plot([g[0], g[1]], [0, 0], color=colors[0], lw=5)
            plt.axvline(g[0], color=colors[0])
            plt.axvline(g[1], color=colors[0], ls='--')
    for ig, gti in enumerate(gti_list):
        mask = mp_create_gti_mask(times, gti, verbose=0)
        mask0 = np.logical_and(mask0, mask)
        if debug:
            for g in gti:
                plt.plot([g[0], g[1]], [ig * step, ig * step],
                         color=colors[ig], lw=5)
                plt.axvline(g[0], color=colors[ig])
                plt.axvline(g[1], color=colors[ig], ls='--')

    gtis = mp_create_gti_from_condition(times, mask0)
    if debug:
        ig += 1
        for g in gti:
            plt.plot([g[0], g[1]], [ig * step, ig * step],
                     color=colors[ig], lw=5)
            plt.axvline(g[0], color=colors[ig])
            plt.axvline(g[1], color=colors[ig], ls='--')

        plt.ylim([-10, 10])
        plt.show()
    return gtis
